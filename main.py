import os
import gc
import math
import json
import random
import resource
import logging
import argparse
import datetime
import time
import pynvml
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.tensorboard import SummaryWriter

import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "social_rec")))
from common_preprocess.metrics import rmse as _rmse, mae as _mae, dump_predictions  # noqa: E402

import data_making as dm
from utils import *
from config import Config
from dataset import EncoderDataset, DecoderDataset, MyDataset
from models.transformer import Transformer
from scheduler import WarmupCosineSchedule

# hyperparmeter tuning
import ray
from ray import tune
from ray.air import session
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

logger = logging.getLogger(__name__)

class DeviceError(Exception):
    def __init__(self):
        super().__init__("GPU not Available.")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_encoder(device, model, optimizer, lr_scheduler, ds_iter, training_config):
    # ######################### 학습 전, attention map 저장 #########################
    tmp_batch = next(iter(ds_iter['train']))
    tmp_batch = {k:v.to(device) for k,v in tmp_batch.items()}
    model.eval()
    with torch.no_grad():
        enc_output, global_preference = model.encoder(tmp_batch)
        torch.save(model.encoder.global_attention.cpu(), 'soft_attn_b4.pt') # attention map 저장
    # ############################################################################

    # TODO: Epoch당 loss, RMSE, MAE 추적 => TensorBoard 또는 파일 저장을 통해 tracing할 수 있도록.
    logger.info("***** Running Encoder training *****")
    logger.info("Total steps = %d", len(ds_iter['train_enc']))

    best_rmse = 9999.0
    best_mae = 9999.0

    checkpoint_path = training_config['enc_checkpoint_path']
    total_epochs = training_config["num_epochs"]
    enc_select = training_config.get('enc_select', True)
    print(total_epochs)

    update_cnt = 0
    model.train()
    metrics = Metrics()
    # Training step
    for epoch in range(total_epochs):
    # for epoch in range(100):
        sub_losses = AverageMeter()
        # PERF: accumulate the running loss on-GPU and sync (.item()) only once per epoch.
        run_loss = torch.zeros((), device=device)
        # encoder 학습
        enc_iterator = tqdm(ds_iter['train_enc'], desc="Encoder (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(enc_iterator):
            batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            # forward pass
            enc_output, global_preference = model.encoder(batch)

            sub_mask = (batch['item_rating'] != 0)

            # RMSE
            sub_loss = metrics.RMSE(global_preference, batch['item_rating'], sub_mask)

            # Huber Loss
            # sub_loss = F.huber_loss(global_preference, batch['item_rating'], delta=0.1, reduction='none')

            run_loss += sub_loss.detach()

            # NOTE: the old nn.utils.clip_grad_value_(...) call sat BEFORE zero_grad()/backward()
            # so it clipped nothing (grads were wiped right after) -- it was a pure ~100
            # tiny-kernel-launch/step cost with no effect. Removed; numerics unchanged.
            optimizer.zero_grad(set_to_none=True)
            sub_loss.backward()
            optimizer.step()
            if step % 50 == 0:
                enc_iterator.set_description(
                            "Encoder Training (%d / %d Steps)" % (step, len(enc_iterator)))
        sub_losses.update((run_loss / max(step + 1, 1)).item(), n=step + 1)

            
        valid_loss, best_rmse, valid_rmse, update_cnt = valid_encoder(device, model, ds_iter, epoch, checkpoint_path, best_rmse, update_cnt, enc_select=enc_select)
        _type, scheduler = lr_scheduler
        if _type=='rp':
            scheduler.step(valid_rmse)
        else:
            scheduler.step()
            print(scheduler.get_lr())

        print(f"Epoch {epoch:03d} || Encoder Loss: {sub_losses.avg:.4f} || Valid Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || best RMSE: {best_rmse:.4f} ||\n")
        if enc_select and update_cnt==30:
            break

    if not enc_select:
        # ablation: no valid-based selection -- keep the last-epoch weights.
        torch.save({"model_state_dict":model.encoder.state_dict()}, checkpoint_path)
        print(f'\t enc_select=False: saved last-epoch encoder weights')

    return best_rmse

def valid_encoder(device, model, ds_iter, epoch, checkpoint_path, best_rmse, update_cnt, enc_select=True):
    eval_losses = AverageMeter()
    model.eval()

    metrics = Metrics()
    total_rmse = 0.0
    preds, targets, masks = [],[],[]
    with torch.no_grad():
        dec_iterator = tqdm(ds_iter['valid_enc'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dec_iterator):     
            batch = {k:v.to(device) for k,v in batch.items()}
            
            enc_output, global_preference = model.encoder(batch)
            
            mask = (batch['item_rating'] != 0)
            masks.append(mask)
            preds.append(global_preference)
            targets.append(batch['item_rating'])
            
            rmse = metrics.RMSE(global_preference, batch['item_rating'], mask).item()
            eval_losses.update(rmse)
            
            
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    total_rmse = metrics.RMSE(preds, targets, masks)

    if not enc_select:
        # ablation: skip valid-based checkpointing / early-stop bookkeeping entirely.
        return eval_losses.avg, best_rmse, total_rmse, update_cnt

    if total_rmse < best_rmse:
        best_rmse = total_rmse
        torch.save({"model_state_dict":model.encoder.state_dict()}, checkpoint_path)
        print(f'\t best model saved: epoch = {epoch}, valid RMSE = {total_rmse:.6f}')
        update_cnt = 0
    else:
        update_cnt += 1

    return eval_losses.avg, best_rmse, total_rmse, update_cnt

def train(device, model, optimizer, lr_scheduler, ds_iter, training_config, alpha=6.5):

    # TODO: Epoch당 loss, RMSE, MAE 추적 => TensorBoard 또는 파일 저장을 통해 tracing할 수 있도록.
    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", len(ds_iter['train']))

    best_rmse = 9999.0
    best_mae = 9999.0
    best_ndcg = 0
    best_model = None

    checkpoint_path = training_config['checkpoint_path']
    total_epochs = training_config["num_epochs"]

    model.train()
    init_t = time.time()
    total_time = 0
    update_cnt = 0
        
    lr_lst = []
    metrics = Metrics()
    for epoch in range(total_epochs):
        losses = AverageMeter()
        main_losses = AverageMeter()
        rank_losses = AverageMeter()
        # PERF: accumulate running losses on-GPU, sync (.item()) once per epoch.
        run_loss = torch.zeros((), device=device)
        run_main = torch.zeros((), device=device)
        run_rank = torch.zeros((), device=device)
        # decoder 학습
        dec_iterator = tqdm(ds_iter['train'], desc="Decoder (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dec_iterator):
            batch = {k:v.to(device, non_blocking=True) for k,v in batch.items()}
            # forward pass
            output, global_preference = model(batch)

            main_mask = (batch['anchor_ratings'] != 0)

            # # MSE
            # main_loss = metrics.MSE(output, batch['anchor_ratings'], main_mask)

            # Huber loss
            main_loss = F.huber_loss(output[main_mask], batch['anchor_ratings'][main_mask], delta=training_config.get('huber_delta', 1.5))

            # Rank loss mask 생성
            fill_value = -1e9
            masked_output = output.masked_fill(~main_mask, fill_value)
            masked_target = batch['anchor_ratings'].masked_fill(~main_mask, fill_value)

            # KLD (target에 temperature를 적용해 분포를 완화 -> train에서 rank_loss가 0으로 붕괴하는 것을 방지)
            rank_temperature = training_config.get('rank_temperature', 2.0)
            rank_logit = F.log_softmax(masked_output, dim=-1).float()
            rank_target = F.softmax(masked_target / rank_temperature, dim=-1)
            rank_loss = F.kl_div(rank_logit, rank_target, reduction='batchmean')

            # # InfoNCE 스타일의 Rank Loss
            # rank_target_idx = torch.argmax(batch['anchor_ratings'], dim=-1)
            # rank_loss = F.cross_entropy(masked_output / 0.5, rank_target_idx)

            loss = main_loss + rank_loss*alpha
            # loss = main_loss

            run_loss += loss.detach()
            run_main += main_loss.detach()
            run_rank += rank_loss.detach()

            # NOTE: old clip_grad_value_ ran before zero_grad()/backward() -> clipped nothing,
            # only cost ~100 tiny kernel launches/step. Removed; numerics unchanged.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                dec_iterator.set_description(
                            "Decoder Training (%d / %d Steps)" % (step, len(dec_iterator)))
        _n = step + 1
        losses.update((run_loss / _n).item(), n=_n)
        main_losses.update((run_main / _n).item(), n=_n)
        rank_losses.update((run_rank / _n).item(), n=_n)

        # total_time += (start.elapsed_time(end))
        valid_loss, best_rmse, best_ndcg, valid_ndcg, valid_rmse, update_cnt, best_model = valid(device, model, ds_iter, epoch, checkpoint_path, step, best_rmse, best_ndcg, update_cnt, best_model)
        _type, scheduler = lr_scheduler
        if _type=='rp':
            scheduler.step(valid_rmse)
        else:
            scheduler.step()
            print(scheduler.get_lr())
            lr_lst.extend(scheduler.get_lr())

        print(f"Epoch {epoch:03d}: Main Loss: {main_losses.avg:.4f} || Rank Loss: {rank_losses.avg:.4f} || Valid Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || best RMSE: {best_rmse:.4f} ||\n")
        if update_cnt > 30: 
            break

    print('\n [Train Finished]')
    print("total training time (s): {}".format((time.time()-init_t)))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("total memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=device.index))
        
    return best_model
        

def valid(device, model, ds_iter, epoch, checkpoint_path, global_step, best_rmse, best_ndcg, update_cnt, best_model):
    eval_losses = AverageMeter()
    metrics = Metrics()
    output_df = pd.DataFrame()
    total_rmse, total_mae = 0.0, 0.0
    model.eval()
    preds, targets, masks = [],[],[]
    with torch.no_grad():
        
        dec_iterator = tqdm(ds_iter['valid'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dec_iterator):     
            batch = {k:v.to(device) for k,v in batch.items()}
            
            output, global_preference = model(batch)
            mask = (batch['anchor_ratings'] != 0)
            
            masks.append(mask)
            preds.append(output)
            targets.append(batch['anchor_ratings'])
            
            rmse = metrics.RMSE(output, batch['anchor_ratings'], mask).item()

            eval_losses.update(rmse)

            dec_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(dec_iterator), rmse))

            # --- [implicit/ranking] per-user 출력 수집: NDCG용. explicit(RMSE/MAE) 실험에서는 미사용. ---
            # # zero padding인 경우 제외
            # mask = (batch['anchor_items']!=0)
            # output = output*mask
            #
            # item_lst, rating_lst, output_lst = [],[],[]
            # for i in range(batch['anchor_user'].size(0)):
            #     mask = (batch['anchor_items'][i]!=0) # padding되지 않은 index
            #     items = batch['anchor_items'][i][mask].data.cpu().tolist()
            #     ratings = batch['anchor_ratings'][i][mask].data.cpu().tolist()
            #     outputs = output[i][mask].data.cpu().tolist()
            #     assert len(items)==len(ratings)==len(outputs)
            #     item_lst.append(items)
            #     rating_lst.append(ratings)
            #     output_lst.append(outputs)
            # anchor_users = batch['anchor_user'].data.cpu().tolist()
            #
            # df = pd.DataFrame({'anchor_user':anchor_users,
            #                    'anchor_items':item_lst,
            #                    'anchor_ratings':rating_lst,
            #                    'outputs':output_lst})
            #
            # output_df = pd.concat([output_df, df])

    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    total_rmse = metrics.RMSE(preds, targets, masks)
    total_mae = metrics.MAE(preds, targets, masks)

    # --- [implicit/ranking] NDCG 계산: explicit 실험에서는 미사용 (지우지 말 것) ---
    # output_df = output_df.groupby('anchor_user').agg({'anchor_items':lambda x:sum(x, start=[]),
    #                                                       'anchor_ratings':lambda x:sum(x, start=[]),
    #                                                       'outputs':lambda x:sum(x, start=[])}).reset_index()
    # output_df['logits'] = output_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1))
    # output_df['targets'] = output_df['anchor_ratings'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1))
    # output_df['ndcg'] = output_df.apply(lambda x:metrics.NDCG(x['anchor_items'], x['logits'], x['anchor_ratings'], 10), axis=1)
    # output_df = output_df.dropna(how='any')
    # total_ndcg = output_df['ndcg'].mean()
    total_ndcg = 0.0

    # 모델 선택 기준: valid RMSE (explicit)
    if best_rmse > total_rmse:
        best_ndcg = total_ndcg
        best_rmse = total_rmse
        best_model = model.state_dict()
        torch.save({"model_state_dict":best_model}, checkpoint_path)
        print(f'\t best model saved: step = {global_step}, epoch = {epoch}, valid RMSE = {total_rmse:.6f}, valid MAE = {total_mae:.6f}')
        update_cnt = 0
    else:
        update_cnt += 1

    return eval_losses.avg, best_rmse, best_ndcg, total_ndcg, total_rmse, update_cnt, best_model
    
def eval(device, model, ds_iter, split='test', dataset=None, seed=None):
    model.eval()
    metrics = Metrics()
    dump_u, dump_i, dump_t, dump_p = [], [], [], []
    # if device.type=='cuda':
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     stream = torch.cuda.current_stream(device=device)
    #     start.record(stream)

    epoch_iterator = tqdm(ds_iter[split],
                    desc="Validating (X / X Steps) (loss=X.X)",
                    ascii=" =",
                    bar_format="{l_bar}{r_bar}",
                    dynamic_ncols=True,
                    leave=False)

    total_rmse, total_mae = 0.0, 0.0
    preds, targets, masks = [],[],[]
    output_df = pd.DataFrame()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):     
            batch = {k:v.to(device) for k,v in batch.items()}
            
            output, global_preference = model(batch)
            
            mask = (batch['anchor_ratings'] != 0)
            masks.append(mask)
            preds.append(output)
            targets.append(batch['anchor_ratings'])
            rmse = metrics.RMSE(output, batch['anchor_ratings'], mask).item()

            # flat (user, item, y_true, y_pred) for the shared prediction dump
            _uid = batch['anchor_user']
            if _uid.dim() < batch['anchor_items'].dim():
                _uid = _uid.unsqueeze(-1).expand_as(batch['anchor_items'])
            _m = mask.bool()
            dump_u.extend(_uid[_m].data.cpu().tolist())
            dump_i.extend(batch['anchor_items'][_m].data.cpu().tolist())
            dump_t.extend(batch['anchor_ratings'][_m].data.cpu().tolist())
            dump_p.extend(output[_m].data.cpu().tolist())
            
            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), rmse))

            # --- [implicit/ranking] per-user 출력 수집: NDCG/Precision/HR용. explicit 실험에서는 미사용 ---
            # # zero padding인 경우 제외
            # mask = (batch['anchor_items']!=0)
            #
            # item_lst, rating_lst, output_lst = [],[],[]
            # for i in range(batch['anchor_user'].size(0)): # batch내에서 user별 iteration
            #     mask = (batch['anchor_items'][i]!=0) # padding되지 않은 index
            #     items = batch['anchor_items'][i][mask].data.cpu().tolist()
            #     ratings = batch['anchor_ratings'][i][mask].data.cpu().tolist()
            #     outputs = output[i][mask].data.cpu().tolist()
            #     assert len(items)==len(ratings)==len(outputs)
            #     item_lst.append(items)
            #     rating_lst.append(ratings)
            #     output_lst.append(outputs)
            # anchor_users = batch['anchor_user'].data.cpu().tolist()
            #
            # df = pd.DataFrame({'anchor_user':anchor_users,
            #                 'anchor_items':item_lst,
            #                 'anchor_ratings':rating_lst,
            #                 'outputs':output_lst})
            #
            # output_df = pd.concat([output_df, df])
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    # shared benchmark metrics (same RMSE/MAE formula as every other baseline)
    total_rmse = _rmse(dump_t, dump_p)
    total_mae = _mae(dump_t, dump_p)
    if split == 'test' and dataset is not None:
        dump_predictions("soft", dataset, seed, dump_u, dump_i, dump_t, dump_p)

    # --- [implicit/ranking] NDCG@10 / Precision@10 / HR@10: explicit 실험에서는 미사용 (지우지 말 것) ---
    # output_df = output_df.groupby('anchor_user').agg({'anchor_items':lambda x:sum(x.tolist(), start=[]),
    #                                                     'anchor_ratings':lambda x:sum(x.tolist(), start=[]),
    #                                                     'outputs':lambda x:sum(x.tolist(), start=[])}).reset_index()
    # non_neg_df = output_df.apply(lambda x:filter_negs(x), axis=1)
    #
    # output_df['logits'] = output_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    # non_neg_df['logits'] = non_neg_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    # # negative sampling 포함
    # neg_ = output_df.apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    # neg_ = neg_.dropna(how='any')
    # neg_ndcg = neg_['ndcg@10'].mean()
    # neg_precision = neg_['precision@10'].mean()
    # neg_hr = neg_['hr@10'].mean()
    # # negative sampling 미포함 + item 개수별 filter x
    # non_neg = non_neg_df.apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    # non_neg = non_neg.dropna(how='any')
    # non_neg_ndcg = non_neg['ndcg@10'].mean()
    # non_neg_precision = non_neg['precision@10'].mean()
    # # non neg + item 개수별 filter o
    # filtered = non_neg_df[non_neg_df['anchor_items'].apply(len)>=10].apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    # filtered_ndcg = filtered['ndcg@10'].mean()
    # filtered_precision = filtered['precision@10'].mean()

    print("\n [Evaluation Results] (split=%s)" % split)
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print("################################")
    # print("NEG NDCG@10: %2.5f" % neg_ndcg)          # [implicit] 미사용
    # print("NEG PRECISION@10: %2.5f" % neg_precision) # [implicit] 미사용
    # print("NEG HR@10: %2.5f" % neg_hr)               # [implicit] 미사용
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))

    return total_rmse, total_mae
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_total_ram_gb():
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return (pages * page_size) / (1024 ** 3)

def get_peak_ram_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)

def get_args():
                
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--encoder", action='store_true')
    parser.add_argument("--decoder", action='store_true')
    parser.add_argument("--moe", action='store_false')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--tune", action='store_true')
    parser.add_argument("--adaptive_tune", action='store_true')
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=4.0)
    parser.add_argument("--num_tunes", type=int, default=50)
    parser.add_argument("--time_budget", type=int, default=28800,
                        help="Ray Tune wall-clock budget in seconds (default 8h)")
    parser.add_argument("--enc_select", type=str2bool, default=True,
                        help="True: valid-RMSE 기반 best encoder ckpt 저장/early-stop. "
                             "False (ablation): 선택 없이 num_epochs 학습 후 마지막 가중치 사용")
    parser.add_argument("--config_json", type=str, default=None,
                        help="non-tune 실행 시 HP dict(JSON)를 읽어 Ray config처럼 merge")
    parser.add_argument("--sweep_bs", type=int, default=None,
                        help="Ray Tune sweep 시 bs_enc=bs_dec 를 이 값으로 고정. "
                             "설정 시 large-batch에 맞춰 LR 탐색 범위를 상향한다.")
    parser.add_argument("--sweep_epochs", type=int, default=None,
                        help="Ray Tune sweep 의 num_epochs cap 재정의(기본: 데이터셋별 값)")
    parser.add_argument("--tune_concurrency", type=int, default=1,
                        help="동시에 돌릴 trial 수 (fractional GPU 공유). 작은 config가 GPU 메모리를 "
                             "일부만 쓸 때 처리량↑. --gpu_per_trial 와 함께 사용.")
    parser.add_argument("--gpu_per_trial", type=float, default=1.0,
                        help="trial당 GPU 분수 (예: 0.33 -> GPU 1개에서 최대 3 trial 동시)")
    parser.add_argument("--sweep_heads", type=str, default=None,
                        help="num_heads 탐색 후보를 콤마리스트로 재정의 (예: '4' 또는 '4,6'). "
                             "--sweep_bs 와 함께 large-batch에서 작은 모델만 볼 때 유용.")
    parser.add_argument("--joint_train", action='store_true',
                        help="encoder pre-training(train_encoder) 단계를 생략하고 "
                             "encoder-decoder 를 decoder 목적함수로 end-to-end 학습")
    # dataset args
    parser.add_argument("--dataset", type = str, default="ciao_timestamp", help = "ciao, epinions")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="percentage of valid/test dataset")
    parser.add_argument('--user_seq_len', type=int, default=30, help="user random walk sequence length")
    parser.add_argument('--item_per_user', type=int, default=5, help="number of items per user")
    parser.add_argument('--augs', type=int, default=1, help="how many times augment train data per anchor user")
    parser.add_argument('--regen', type=str, default='no', help="[no, all, rw, total, train]")
    parser.add_argument('--bs', type=int, default=32, help="Batch size of dataloader")
    parser.add_argument('--neg', type=str2bool, default=False)
    parser.add_argument('--drop_cold_eval', type=str2bool, default=False,
                        help="exclude anchors with 0 training interactions from valid/test")

    # tuning
    parser.add_argument('--enc_scheduler', type=str, default='rp')
    parser.add_argument('--dec_scheduler', type=str, default='rp')
    
    args = parser.parse_args()
    return args

def args_to_dict(args):
    """argparse.Namespace를 직렬화 가능한 딕셔너리로 변환"""
    return {
        'encoder': args.encoder,
        'moe': args.moe,
        'device': args.device,
        'id': args.id,
        'eval': args.eval,
        'seed': args.seed,
        'dataset': args.dataset,
        'test_ratio': args.test_ratio,
        'user_seq_len': args.user_seq_len,
        'item_per_user': args.item_per_user,
        'augs': args.augs,
        'regen': args.regen,
        'bs': args.bs,
        'neg': args.neg,
        'drop_cold_eval': getattr(args, 'drop_cold_eval', False),
        'enc_scheduler': args.enc_scheduler,
        'dec_scheduler': args.dec_scheduler,
    }
    
def run(config):
    import torch, random, numpy as np # 함수 내부에서 import 해줘야 함
    
    os.chdir(Path(__file__).parent.resolve())
    # if 'args_dict' in config:
    #     args = argparse.Namespace(**config['args_dict'])
    # else:
    #     args = argparse.Namespace(**config)
        
    if 'args_dict' in config:
        args = argparse.Namespace(**config['args_dict'])
        for k,v in config.items():
            if k=='args_dict':
                continue
            setattr(args, k, v)
    else:
        args = argparse.Namespace(**config)
    #     setattr(args, 'alpha', 6.5)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # regen 여부 확인
    if args.regen == 'all':
        print("Re-Creating All Datatset...")
    elif args.regen=='total':
        print("Re-Creating total_df...")
    elif args.regen=='rw':
        print("Re-Creating rw_seq & total_df...")
    else:
        print("Loading Datatset...")
    data_making = dm.DatasetMaking(args) 
    print("\n")
    
    total_train = data_making.total_train
    total_valid = data_making.total_valid
    total_test = data_making.total_test

    min_item_len = data_making.min_item_len

    # dataset & dataloader
    train_enc = EncoderDataset(total_train)
    valid_enc = EncoderDataset(total_valid)
    test_enc = EncoderDataset(total_test)
    train_ds = DecoderDataset(total_train)
    valid_ds = DecoderDataset(total_valid)
    test_ds = DecoderDataset(total_test)
    
    model_config = Config[args.dataset]["model"].copy()
    training_config = Config[args.dataset]["training"].copy()

    # loss-computation knobs exposed for tuning (not in config.py). Pre-seeding them here
    # lets the Ray `config` merge loop below override them like any other training_config key.
    training_config.setdefault('huber_delta', 1.5)
    training_config.setdefault('rank_temperature', 2.0)
    training_config['enc_select'] = str2bool(getattr(args, 'enc_select', True))

    if config!=None:
        for k,v in config.items():
            if k=='args_dict':
                continue
            elif k in model_config:
                model_config[k]=v
            elif k in training_config:
                training_config[k]=v 
                
    # model config - num users & num items & degrees
    model_config["num_user"] = data_making.num_user
    model_config["num_item"] = data_making.num_item
    model_config['min_item_len'] = min_item_len
    model_config['user_seq_len'] = args.user_seq_len
    model_config['item_seq_len'] = args.user_seq_len*args.item_per_user
    model_config["max_user_degree"] = data_making.max_user_degree
    model_config["max_item_degree"] = data_making.max_item_degree
    model_config['moe'] = str2bool(args.moe)
    
    # PERF: TF32 matmul/conv is opt-in (SOFT_TF32=1) so it stays off for sweep reproducibility
    # but can be enabled for final retrains.
    if os.environ.get('SOFT_TF32', '0') == '1':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    test_bs = int(os.environ.get('SOFT_TEST_BS', 512))
    # datasets are fully preloaded as in-memory tensors. pin_memory + a few persistent
    # workers + prefetch keep the H2D copies off the critical path; SOFT_NUM_WORKERS
    # overrides for unattended sweeps if fork/shm pressure shows up.
    num_workers = int(os.environ.get('SOFT_NUM_WORKERS', 4))
    _train_kw = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        _train_kw.update(persistent_workers=True, prefetch_factor=4)
    _eval_kw = dict(num_workers=min(2, num_workers), pin_memory=True)
    if _eval_kw['num_workers'] > 0:
        _eval_kw.update(persistent_workers=True)
    ds_iter = {
            "train_enc":DataLoader(train_enc, batch_size = training_config['bs_enc'], shuffle=True, **_train_kw),
            "train":DataLoader(train_ds, batch_size = training_config['bs_dec'], shuffle=True, **_train_kw),
            "valid_enc":DataLoader(valid_enc, batch_size = test_bs, shuffle=False, **_eval_kw),
            "valid":DataLoader(valid_ds, batch_size = test_bs, shuffle=False, **_eval_kw),
            "test_enc":DataLoader(test_enc, batch_size = test_bs, shuffle=False, **_eval_kw),
            "test":DataLoader(test_ds, batch_size = test_bs, shuffle=False, **_eval_kw)
    }
    
    ######################################################### model initialization #########################################################

    ### log preparation ###
    log_dir = os.getcwd() + f'/logs/log_seed_{args.seed}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_data = os.getcwd() + f'/checkpoints/{args.dataset}/'
    if not os.path.exists(checkpoint_data):
        os.makedirs(checkpoint_data)
    checkpoint_dir = checkpoint_data + f'checkpoints_seed_{args.seed}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, "train")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    name_moe = 'moe' if model_config['moe'] else 'ffn'
    if model_config['moe']:
        name = ('_').join([str(v) for k,v in model_config.items() if k not in ['moe']])
        name_enc = ('_').join([str(v) for k,v in model_config.items() if k not in ['moe','dec_blocks']])
    else:
        name = ('_').join([str(v) for k,v in model_config.items() if k not in ['n_experts','topk','moe']])
        name_enc = ('_').join([str(v) for k,v in model_config.items() if k not in ['n_experts','topk','moe','dec_blocks']])
        
    name = name+'_'+name_moe
    name_enc = name_enc+'_'+name_moe
                          
    name = name+'_'+('_').join([str(v) for k,v in training_config.items()])
    name_enc = name_enc+'_'+('_').join([str(v) for k,v in training_config.items() if k not in ['weight_decay_dec','lr', 'bs_dec']])
    
    print(model_config)
    print(training_config)
    checkpoint_path = os.path.join(checkpoint_dir, f'{name}_{args.dec_scheduler}.model') # set model name
    checkpoint_enc = os.path.join(checkpoint_dir, f'{name_enc}_{args.enc_scheduler}.model') # set model name
    print(checkpoint_path, "\n")
    print(checkpoint_enc, "\n")
    training_config["checkpoint_path"] = checkpoint_path
    training_config["enc_checkpoint_path"] = checkpoint_enc

    # Device
    pynvml.nvmlInit()
    device_ids = list(range(torch.cuda.device_count()))
    for i in device_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if not util_info.gpu:
            device = torch.device(f'cuda:{i}')
            break
        else:
            device = torch.device('cpu')

    if args.device=='cpu':
        device = torch.device(args.device)
    else:
        device = torch.device(f'cuda:{args.id}' if torch.cuda.is_available() else 'cpu')
        # device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    # print(f"GPU index: {device.index}")
    print(f"GPU : {device}")
    print("\n")
    pynvml.nvmlShutdown()
        
    ### model preparation ###
    model = Transformer(**model_config).to(device)
    if not args.eval:
        joint = bool(getattr(args, 'joint_train', False))
        opt_enc = None
        enc_rmse = torch.tensor(float('nan'))

        if joint:
            # encoder pre-training 생략 -> opt_dec(model.parameters())가 encoder+decoder를 함께 학습
            print("\n[joint_train] encoder pre-training(train_encoder) 생략 -> encoder-decoder end-to-end 학습\n")
        else:
            # Encoder
            # user/item ID 임베딩(node_encoder)은 특정 id에 과적합(암기)되기 쉬우므로 별도로 더 강한 weight decay를 적용
            id_embed_params = [p for n, p in model.encoder.named_parameters() if n.endswith('node_encoder.weight')]
            other_params = [p for n, p in model.encoder.named_parameters() if not n.endswith('node_encoder.weight')]
            id_embed_weight_decay = training_config['weight_decay_enc'] * 3

            opt_enc = torch.optim.AdamW(
                [
                    {'params': other_params, 'weight_decay': training_config['weight_decay_enc']},
                    {'params': id_embed_params, 'weight_decay': id_embed_weight_decay},
                ],
                lr = training_config['lr_enc'],
                betas=[0.9,0.999])

            if args.enc_scheduler=='rp':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = opt_enc,
                mode = 'min',
                factor = 0.9,
                patience = 3,
                min_lr=1e-5,
                threshold = 1e-3,
                verbose = True
                )
            else:
                scheduler = CosineAnnealingWarmupRestarts(
                optimizer=opt_enc,
                first_cycle_steps=5,
                cycle_mult=2,
                max_lr = training_config['lr_enc'],
                min_lr=1e-5,
                warmup_steps=1,
                gamma=0.9,
                )

            lr_scheduler = (args.enc_scheduler, scheduler)

            # Encoder Train
            if not os.path.isfile(training_config['enc_checkpoint_path']) or args.encoder:
                enc_rmse = train_encoder(device, model, opt_enc, lr_scheduler, ds_iter, training_config)
            # Encoder Load
            else:
                print("Best Encoder model loaded!")
            checkpoint = torch.load(training_config['enc_checkpoint_path'])
            model.encoder.load_state_dict(checkpoint['model_state_dict'])
            _, _, enc_rmse, _ = valid_encoder(device, model, ds_iter, 0, checkpoint_path, 0, 0)

        if args.decoder:
            # Decoder
            opt_dec = torch.optim.AdamW(
                model.parameters(),
                lr = training_config['lr'],
                betas=[0.9,0.999],
                weight_decay=training_config['weight_decay_dec'])
                        
            if args.dec_scheduler=='rp':
                # decoder 초기 LR 재설정
                for param_group in opt_dec.param_groups:
                    param_group['lr'] = training_config['lr']

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = opt_dec,
                mode = 'min',
                factor = 0.9,
                patience = 3,
                min_lr=1e-5,
                threshold = 1e-3,
                verbose = True
                )
            else:
                scheduler = CosineAnnealingWarmupRestarts(
                optimizer=opt_dec,
                first_cycle_steps=300,
                cycle_mult=1,
                max_lr = training_config['lr'],
                min_lr=1e-5,
                warmup_steps=0,
                gamma=1,
                )
                
            lr_scheduler = (args.dec_scheduler, scheduler)
            
            # Decoder Train
            # model.load_state_dict(best_model)
            _ = train(device, model, opt_dec, lr_scheduler, ds_iter, training_config, args.alpha)
            checkpoint = torch.load(training_config['checkpoint_path'])
            model.load_state_dict(checkpoint['model_state_dict'])
            # 모델 선택은 valid로 끝났고, 여기서 valid/test를 각각 1회 측정한다 (explicit: RMSE/MAE).
            # Ray 튜닝 목적값 `score`는 valid RMSE 기준(아래 session.report), test_*는 보고용.
            val_rmse, val_mae = eval(device, model, ds_iter, split='valid')
            test_rmse, test_mae = eval(device, model, ds_iter, split='test',
                                       dataset=getattr(args, 'dataset', None),
                                       seed=getattr(args, 'seed', None))

            if joint:
                # joint 학습된 encoder 자체의 preference-head valid RMSE (참고용, 저장 안 함)
                _, _, enc_rmse, _ = valid_encoder(device, model, ds_iter, 0, checkpoint_path, 9999.0, 0, enc_select=False)

            # 메모리 정리
            del model, opt_dec
            if opt_enc is not None:
                del opt_enc
            torch.cuda.empty_cache()
            gc.collect()

    else:
        ### eval ###
        checkpoint_path = training_config['checkpoint_path']
        print(checkpoint_path)
        if os.path.exists(checkpoint_path): #and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
            model.eval()
            # user_idx = []
            # user_reptn = []
            with torch.no_grad():
                batch = next(iter(ds_iter['train']))
                batch = {k:v.to(device) for k,v in batch.items()}
                # user_idx.append(batch['user_seq'].cpu())
                
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                print("loading the best model from: " + checkpoint_path)
                test_rmse, test_mae = eval(device, model, ds_iter, split='test',
                                           dataset=getattr(args, 'dataset', None),
                                           seed=getattr(args, 'seed', None))
                _ = model(batch)
                torch.save(model.encoder.global_attention.cpu(), 'soft_attn.pt') # attention map 저장
                # user_reptn.append(model.decoder.user_reptn.cpu())
                
            # user_idx = torch.cat(user_idx, dim=0)
            # user_reptn = torch.cat(user_reptn, dim=0)
            # torch.save(user_idx, 'user_idx_soft.pt') # user representation(embedding) 저장
            # torch.save(user_reptn, 'user_embed_soft.pt') # user representation(embedding) 저장
            
                torch.save(model.user_embed.node_encoder.weight.cpu(), 'user_embed_soft.pt') # user representation(embedding) 저장
        else:
            print("No Best Model Found")
            
        print(model_config)
        training_config = {k:v for k,v in training_config.items() if k not in ['checkpoint_path', 'enc_checkpoint_path']}
        print(training_config)
    
    if args.tune:
        peak_ram_gb = get_peak_ram_gb()
        
        if args.decoder:
            # explicit 실험: 목적값 score = 1/valid_RMSE (mode='max' 그대로). test_*는 보고용.
            session.report({
                            'enc_rmse':   enc_rmse.cpu().item(),
                            'score':      float(1.0 / val_rmse),           # Ray metric (mode='max') -- valid RMSE 기준 선택
                            'rmse':       float(val_rmse),                 # 기존 출력 코드 호환용 alias (= valid)
                            'val_rmse':   float(val_rmse),
                            'val_mae':    float(val_mae),
                            'test_rmse':  float(test_rmse),                # 최종 보고용 (trial 선택에는 미사용)
                            'test_mae':   float(test_mae),
                            'peak_ram_gb': peak_ram_gb
                            })
        else:
            session.report({
                            'enc_rmse': enc_rmse.cpu().item(),
                            'peak_ram_gb': peak_ram_gb
                            })

def main():
    args = get_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args_dict = vars(args)
    ######################################################### data preparation #########################################################    
    if args.tune:
        print("="*60)
        print("Starting Ray Tune Hyperparameter Search")
        print("="*60)
        print(f"Fixed args: {args_dict}")
        # parameters to Tune
        # NOTE: 데이터 재생성이 필요한 파라미터(user_seq_len / item_per_user / neg / dataset /
        # test_ratio)는 여기에 두지 않는다 -- args_dict의 고정값을 그대로 사용.
        # bs_enc/bs_dec도 단일 GPU OOM 회피를 위해 고정(config.py 기본).
        #
        # 데이터셋별 sweep 설정: 큰 데이터셋(epinions: interactions 2.7x, items 2.5x)은
        # trial당 시간이 ~2.5x라 10h budget 안에서 trial 수 확보 + 12GB OOM 회피를 위해
        # sweep epoch을 줄이고 d_model 512(num_heads=8)를 제외한다.
        # (최종 best는 어느 경우든 num_epochs=300 으로 재학습)
        if args.dataset == 'epinions':
            sweep_epochs  = 70
            heads_choice  = [4, 6]        # d_model 256 / 384
        else:
            sweep_epochs  = 100
            heads_choice  = [4, 6, 8]     # d_model 256 / 384 / 512

        search_space = {
            'args_dict': args_dict,
            # --- optimizer ---
            'lr_enc':           tune.loguniform(3e-4, 1e-2),
            'lr':               tune.loguniform(1e-4, 3e-3),
            'weight_decay_enc': tune.loguniform(1e-3, 1.5e-1),
            'weight_decay_dec': tune.loguniform(1e-3, 1.5e-1),
            # --- architecture (d_model = num_heads*64, topk < n_experts) ---
            # ranges kept within ~10GB so trials fit the single 12GB RTX 3080 Ti.
            'dropout':    tune.choice([0.1, 0.2, 0.3, 0.4]),
            'num_heads':  tune.choice(heads_choice),
            'd_model':    tune.sample_from(lambda spec: spec.config['num_heads'] * 64),
            'd_ffn':      tune.choice([256, 512]),
            'enc_blocks': tune.choice([1, 2]),
            'dec_blocks': tune.choice([1, 2, 3]),
            'n_experts':  tune.choice([4, 6]),
            'topk':       tune.sample_from(lambda spec: random.randint(2, spec.config['n_experts'] - 1)),
            # --- loss ---
            'alpha':            tune.uniform(1.0, 8.0),
            'huber_delta':      tune.choice([0.5, 1.0, 1.5, 2.0]),
            'rank_temperature': tune.choice([1.0, 1.5, 2.0, 3.0]),
            # sweep 동안만 epoch 축소 (최종 숫자는 best config로 별도 재학습)
            'num_epochs': sweep_epochs,
            # 공정 비교를 위해 seed 고정
            'seed': args.seed,
        }

        # --sweep_epochs 로 num_epochs cap 재정의
        if getattr(args, 'sweep_epochs', None):
            search_space['num_epochs'] = int(args.sweep_epochs)

        # --sweep_bs: batch size 고정 + large-batch에 맞춰 LR 탐색 범위 상향.
        # (§4 실험: bs를 키우면 LR 고정 시 RMSE가 단조 악화 -> optimizer HP 재탐색이 필요)
        _sbs = getattr(args, 'sweep_bs', None)
        if _sbs:
            search_space['bs_enc'] = int(_sbs)
            search_space['bs_dec'] = int(_sbs)
            # large batch에서 d_model=512는 (a) trial당 GPU 메모리가 커서 concurrency를 막고
            # (b) §4에서 large-batch 성능 악화가 가장 심했음 -> 기본 256/384 로 제한.
            # --sweep_heads 로 더 좁힐 수 있음 (bs256 sweep 1~2h 관찰: nh6/d384가 OOM 유발 +
            # 성능도 nh4/d256보다 일관되게 나쁨 -> '4' 로 고정하면 OOM 제거 + budget 집중).
            _heads = getattr(args, 'sweep_heads', None)
            if _heads:
                search_space['num_heads'] = tune.choice([int(x) for x in _heads.split(',') if x.strip()])
            else:
                search_space['num_heads'] = tune.choice([4, 6])
            # linear scaling rule 근처: bs 32 최적(lr_enc~4.7e-3, lr~7.5e-4) 대비 상한을 끌어올림.
            _k = _sbs / 32.0
            search_space['lr_enc'] = tune.loguniform(3e-4 * min(_k, 4), 1e-2 * min(_k, 5))
            search_space['lr']     = tune.loguniform(1e-4 * min(_k, 4), 3e-3 * min(_k, 5))
            print(f"[sweep_bs] bs_enc=bs_dec={_sbs} 고정 | "
                  f"lr_enc∈[{3e-4*min(_k,4):.2e},{1e-2*min(_k,5):.2e}] "
                  f"lr∈[{1e-4*min(_k,4):.2e},{3e-3*min(_k,5):.2e}]")

        # ray 초기화 및 실행
        # if args.dataset in ['ciao_timestamp','epinions']:
        #     num_cpus=30
        # else:
        #     num_cpus=15
            
        num_cpus = 14
        if not ray.is_initialized():
            gpu_idx = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            num_gpus = len([x for x in gpu_idx.split(',') if x != ''])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                object_store_memory=8 * 1024**3,
                include_dashboard=False,   # dashboard import는 이 env에서 깨짐 (pkg_resources.packaging)
                ignore_reinit_error=True
            )

        if args.decoder:
            metric = 'score'
            mode = 'max'
        else:
            metric = 'enc_rmse'
            mode = 'min'

        part = 'full' if args.decoder else 'enc'
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d_%H%M")
        storage_path = os.path.join(os.getcwd(), 'ray_results')
        
        if args.adaptive_tune:
            # pilot run을 통해서 trial 당 peak RAM 추정
            estimated_trial_ram_gb = 16
            pilot_samples = 2
            pilot_analysis = tune.run(
                run, config=search_space,
                num_samples = pilot_samples,
                resources_per_trial={'cpu':4, 'gpu':1},
                max_concurrent_trials=1,
                raise_on_failed_trial=False,
                checkpoint_freq=0,
                storage_path=storage_path,
                name=f'{args.dataset}_{part}_{now_str}_pilot',
                metric=metric,
                mode=mode
            )
            
            peak_list = []
            for t in pilot_analysis.trials:
                v = t.last_result.get('peak_ram_gb')
                if v is not None and v>0:
                    peak_list.append(float(v))
                    
            if len(peak_list)>0:
                estimated_trial_ram_gb = max(peak_list)
                
            total_ram_gb = get_total_ram_gb()
            safe_ram_gb = total_ram_gb * 0.8
            max_by_ram = max(1, int(safe_ram_gb // estimated_trial_ram_gb))
            max_by_gpu = 4
            max_by_cpu_floor = num_cpus
            max_concurrent_trials = max(1, min(max_by_ram, max_by_gpu, max_by_cpu_floor))
            adaptive_cpu_per_trial = max(1, int(num_cpus // max_concurrent_trials))
            
            resource_per_trial = {
                'cpu':adaptive_cpu_per_trial,
                'gpu':1
            }
        
            analysis = tune.run(run, config=search_space, num_samples=args.num_tunes,
                            resources_per_trial=resource_per_trial,
                            max_concurrent_trials=max_concurrent_trials,
                            reuse_actors=True,
                            raise_on_failed_trial=False, 
                            checkpoint_freq=0,
                            storage_path=storage_path,
                            name=f'{args.dataset}_{part}_{now_str}',
                            # name=f'{args.dataset}_{part}_seq_20',
                            metric=metric, mode=mode
                            )
        else:
            # 단일 GPU. 기본은 한 번에 한 trial(gpu=1). 작은 config가 GPU 메모리를 일부만
            # 쓰는 경우 --tune_concurrency / --gpu_per_trial 로 fractional GPU 공유 → 처리량↑.
            conc = max(1, int(getattr(args, 'tune_concurrency', 1)))
            gpu_frac = float(getattr(args, 'gpu_per_trial', 1.0))
            if conc > 1 and gpu_frac >= 1.0:
                gpu_frac = round(1.0 / conc, 4)   # 편의: concurrency만 줘도 GPU를 균등 분할
            cpu_per = max(1, num_cpus // conc)
            max_concurrent_trials = conc
            resource_per_trial = {'cpu': cpu_per, 'gpu': gpu_frac}
            print(f"[tune] concurrency={conc} | per-trial cpu={cpu_per} gpu={gpu_frac}")
            analysis = tune.run(run, config=search_space, num_samples=args.num_tunes,
                            time_budget_s=args.time_budget,
                            resources_per_trial=resource_per_trial,
                            max_concurrent_trials=max_concurrent_trials,
                            reuse_actors=True,
                            raise_on_failed_trial=False,
                            max_failures=3,
                            checkpoint_freq=0,
                            storage_path=storage_path,
                            name=f'{args.dataset}_{part}_{now_str}',
                            metric=metric, mode=mode
                            )
                    
        best_trial = analysis.get_best_trial(metric, mode=mode)
        if best_trial is None:
            print("\n[WARN] 완료된 trial이 없습니다. ray_results 로그를 확인하세요.")
            return

        def _hp_only(cfg):
            # args_dict 등 비-HP 키 제거 -> 재학습에 그대로 넘길 수 있는 dict
            return {k: v for k, v in cfg.items() if k not in ('args_dict',)}

        lr = best_trial.last_result
        print(f"\nBest trial: {best_trial.trial_id}")
        print(f"Best config (HP only): {_hp_only(best_trial.config)}")
        print(f"[valid] rmse={lr.get('val_rmse')}  mae={lr.get('val_mae')}  (score=1/rmse={lr.get('score')})")
        print(f"[test ] rmse={lr.get('test_rmse')}  mae={lr.get('test_mae')}")
        print(f"enc_rmse(valid)={lr.get('enc_rmse')}")

        # 재학습용 best config 저장 (num_epochs 제외 -> config.py 300 유지)
        best_hp = _hp_only(best_trial.config)
        best_hp.pop('num_epochs', None)
        out_path = os.path.join(os.getcwd(), f'best_config_{args.dataset}.json')
        with open(out_path, 'w') as f:
            json.dump(best_hp, f, indent=2, default=str)
        print(f"saved -> {out_path}")

        sorted_trials = sorted(analysis.trials,
                               key=lambda t:t.last_result.get(metric, float('-inf') if mode=='max' else float('inf')),
                               reverse=(mode=='max'))

        for t in sorted_trials:
            r = t.last_result
            print(f"Trial: {t.trial_id} | {metric}={r.get(metric)} | "
                  f"val_rmse={r.get('val_rmse')} val_mae={r.get('val_mae')} | "
                  f"test_rmse={r.get('test_rmse')} test_mae={r.get('test_mae')}")
            print(f"  Config: {_hp_only(t.config)}")
            print(f"  Path: {t.local_path}")
            print("-" * 60)
    
    else:
        cfg = dict(args_dict)
        if getattr(args, 'config_json', None):
            with open(args.config_json) as f:
                extra = json.load(f)
            extra.pop('args_dict', None)
            cfg.update(extra)
            print(f"[config_json] merged overrides: {extra}")
        run(config=cfg)

if __name__ == '__main__':
    main()