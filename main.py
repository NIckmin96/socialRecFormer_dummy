import os
import gc
import logging
import argparse
import random
import datetime
import pickle
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
from torch.utils.tensorboard import SummaryWriter

import data_making_2 as dm
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
    print(total_epochs)

    update_cnt = 0
    model.train()
    metrics = Metrics()
    # Training step
    for epoch in range(total_epochs):
    # for epoch in range(100):
        sub_losses = AverageMeter()
        # encoder 학습
        enc_iterator = tqdm(ds_iter['train_enc'], desc="Encoder (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(enc_iterator):
            batch = {k:v.to(device) for k,v in batch.items()}
            # forward pass
            enc_output, global_preference = model.encoder(batch)

            sub_mask = (batch['item_rating'] != 0)
            sub_loss = metrics.RMSE(global_preference, batch['item_rating'], sub_mask)
            sub_losses.update(sub_loss.item())
            
            nn.utils.clip_grad_value_(model.encoder.parameters(), clip_value=1) # Gradient Clipping
            optimizer.zero_grad()            
            sub_loss.backward()
            optimizer.step()
            enc_iterator.set_description(
                        "Encoder Training (%d / %d Steps) (loss=%2.5f)" % (step, len(enc_iterator), sub_losses.avg))
            
            
        valid_loss, best_rmse, valid_rmse, update_cnt = valid_encoder(device, model, ds_iter, epoch, checkpoint_path, best_rmse, update_cnt)
        _type, scheduler = lr_scheduler
        if _type=='rp':
            scheduler.step(valid_rmse)
        else:
            scheduler.step()
            print(scheduler.get_lr())
            
        print(f"Epoch {epoch:03d} || Encoder Loss: {sub_losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || best RMSE: {best_rmse:.4f} ||\n")
        if update_cnt==30: 
            break
        
    return best_rmse

def valid_encoder(device, model, ds_iter, epoch, checkpoint_path, best_rmse, update_cnt):
    eval_losses = AverageMeter()
    model.eval()
    
    metrics = Metrics()
    total_rmse = 0.0
    preds, targets, masks = [],[],[]
    with torch.no_grad():
        # dec_iterator = tqdm(ds_iter['valid'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        dec_iterator = tqdm(ds_iter['test'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
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
            
    if total_rmse < best_rmse: 
        best_rmse = total_rmse
        torch.save({"model_state_dict":model.encoder.state_dict()}, checkpoint_path)
        print(f'\t best model saved: epoch = {epoch}, test RMSE = {total_rmse:.6f}')
        update_cnt = 0
    else:
        update_cnt += 1

    return eval_losses.avg, best_rmse, total_rmse, update_cnt

def train(device, model, optimizer, lr_scheduler, ds_iter, training_config):

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
        # decoder 학습
        dec_iterator = tqdm(ds_iter['train'], desc="Decoder (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dec_iterator):
            batch = {k:v.to(device) for k,v in batch.items()}
            # forward pass
            output, global_preference = model(batch)
            
            main_mask = (batch['anchor_ratings'] != 0)
            main_loss = metrics.MSE(output, batch['anchor_ratings'], main_mask)
            main_losses.update(main_loss.item())

            rank_logit = F.log_softmax(output, dim=-1).float()
            rank_target = F.softmax(batch['anchor_ratings'], dim=-1)
            rank_loss = F.kl_div(rank_logit, rank_target, reduction='batchmean')
            rank_losses.update(rank_loss.item())
            
    
            loss = main_loss + rank_loss
            # loss = main_loss
            losses.update(loss.item())
            
            
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.zero_grad()            
            loss.backward()
            optimizer.step()
            dec_iterator.set_description(
                        "Decoder Training (%d / %d Steps) (loss=%2.5f)" % (step, len(dec_iterator), losses.avg))
            
        # total_time += (start.elapsed_time(end))
        valid_loss, best_rmse, best_ndcg, valid_ndcg, valid_rmse, update_cnt, best_model = valid(device, model, ds_iter, epoch, checkpoint_path, step, best_rmse, best_ndcg, update_cnt, best_model)
        _type, scheduler = lr_scheduler
        if _type=='rp':
            scheduler.step(valid_rmse)
        else:
            scheduler.step()
            print(scheduler.get_lr())
            lr_lst.extend(scheduler.get_lr())

        print(f"Epoch {epoch:03d}: Main Loss: {main_losses.avg:.4f} || Rank Loss: {rank_losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch NDCG@10: {valid_ndcg:.4f} || epoch RMSE: {valid_rmse:.4f} || best RMSE: {best_rmse:.4f} || best NDCG@10: {best_ndcg:.4f} ||\n")
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
        
        # dec_iterator = tqdm(ds_iter['valid'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        dec_iterator = tqdm(ds_iter['test'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
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
            
            # zero padding인 경우 제외
            mask = (batch['anchor_items']!=0)
            output = output*mask
            
            item_lst, rating_lst, output_lst = [],[],[]
            for i in range(batch['anchor_user'].size(0)):
                mask = (batch['anchor_items'][i]!=0) # padding되지 않은 index
                items = items = batch['anchor_items'][i][mask].data.cpu().tolist()
                ratings = batch['anchor_ratings'][i][mask].data.cpu().tolist()
                outputs = output[i][mask].data.cpu().tolist()
                assert len(items)==len(ratings)==len(outputs)
                item_lst.append(items)
                rating_lst.append(ratings)
                output_lst.append(outputs)
            anchor_users = batch['anchor_user'].data.cpu().tolist()
            
            df = pd.DataFrame({'anchor_user':anchor_users,
                               'anchor_items':item_lst,
                               'anchor_ratings':rating_lst,
                               'outputs':output_lst})
            
            output_df = pd.concat([output_df, df])
            
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    total_rmse = metrics.RMSE(preds, targets, masks)
    output_df = output_df.groupby('anchor_user').agg({'anchor_items':lambda x:sum(x, start=[]),
                                                          'anchor_ratings':lambda x:sum(x, start=[]),
                                                          'outputs':lambda x:sum(x, start=[])}).reset_index()
    output_df['logits'] = output_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1))
    output_df['targets'] = output_df['anchor_ratings'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1))    
    output_df['ndcg'] = output_df.apply(lambda x:metrics.NDCG(x['anchor_items'], x['logits'], x['anchor_ratings'], 10), axis=1)
    output_df = output_df.dropna(how='any')
    total_ndcg = output_df[output_df['anchor_items'].apply(len)>=10]['ndcg'].mean()
    
     # 상대 개선 비율 계산
    ndcg_ratio = total_ndcg / best_ndcg if best_ndcg > 0 else 1.0
    rmse_ratio = best_rmse / total_rmse if total_rmse > 0 else 1.0

    # 조건 비교 (score 없이)
    improved = (0.2 * ndcg_ratio + 0.8 * rmse_ratio) > 1.0
    
    if improved:
        best_ndcg = total_ndcg
        best_rmse = total_rmse
        best_model = model.state_dict()
        torch.save({"model_state_dict":best_model}, checkpoint_path)
        print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse:.6f}, test NDCG@10 = {best_ndcg:.6f}')
        update_cnt = 0
    else:
        update_cnt += 1

    return eval_losses.avg, best_rmse, best_ndcg, total_ndcg, total_rmse, update_cnt, best_model
    
def eval(device, model, ds_iter):
    model.eval()
    metrics = Metrics()
    # if device.type=='cuda':
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     stream = torch.cuda.current_stream(device=device)
    #     start.record(stream)
        
    epoch_iterator = tqdm(ds_iter['test'],
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
            
            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), rmse))
            
            # zero padding인 경우 제외
            mask = (batch['anchor_items']!=0)
            
            item_lst, rating_lst, output_lst = [],[],[]
            for i in range(batch['anchor_user'].size(0)): # batch내에서 user별 iteration
                mask = (batch['anchor_items'][i]!=0) # padding되지 않은 index
                items = batch['anchor_items'][i][mask].data.cpu().tolist()
                ratings = batch['anchor_ratings'][i][mask].data.cpu().tolist()
                outputs = output[i][mask].data.cpu().tolist()
                assert len(items)==len(ratings)==len(outputs)
                item_lst.append(items)
                rating_lst.append(ratings)
                output_lst.append(outputs)
            anchor_users = batch['anchor_user'].data.cpu().tolist()
            
            df = pd.DataFrame({'anchor_user':anchor_users,
                            'anchor_items':item_lst,
                            'anchor_ratings':rating_lst,
                            'outputs':output_lst})
            
            output_df = pd.concat([output_df, df])
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    masks = torch.cat(masks, dim=0)
    total_rmse = metrics.RMSE(preds, targets, masks)
    total_mae = metrics.MAE(preds, targets, masks)
            
    output_df = output_df.groupby('anchor_user').agg({'anchor_items':lambda x:sum(x.tolist(), start=[]),
                                                        'anchor_ratings':lambda x:sum(x.tolist(), start=[]),
                                                        'outputs':lambda x:sum(x.tolist(), start=[])}).reset_index()
    non_neg_df = output_df.apply(lambda x:filter_negs(x), axis=1)
    
    output_df['logits'] = output_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    # output_df['targets'] = output_df['anchor_ratings'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    
    non_neg_df['logits'] = non_neg_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    # non_neg_df['targets'] = non_neg_df['anchor_ratings'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    # negative sampling 포함
    neg_ = output_df.apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    neg_ = neg_.dropna(how='any')
    neg_ndcg = neg_['ndcg@10'].mean()
    neg_precision = neg_['precision@10'].mean()
    # neg_recall = neg_['recall@10'].mean()
    neg_hr = neg_['hr@10'].mean()
    # negative sampling 미포함 + item 개수별 filter x
    non_neg = non_neg_df.apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    non_neg = non_neg.dropna(how='any')
    non_neg_ndcg = non_neg['ndcg@10'].mean()
    non_neg_precision = non_neg['precision@10'].mean()
    # non_neg_recall = non_neg['recall@10'].mean()
    # non neg + item 개수별 filter o
    filtered = non_neg_df[non_neg_df['anchor_items'].apply(len)>=10].apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    filtered_ndcg = filtered['ndcg@10'].mean()
    filtered_precision = filtered['precision@10'].mean()          
    
    # neg_.to_csv(f'eval_output_{args.dataset}_{args.item_per_user}_neg.csv', index=False)
    # non_neg.to_csv(f'eval_output_{args.dataset}_{args.item_per_user}_non_neg.csv', index=False)
        
    # if device.type=='cuda':
    #     end.record(stream)
    #     torch.cuda.synchronize()

    print("\n [Evaluation Results]")
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print("################################")
    print("NEG NDCG@10: %2.5f" % neg_ndcg)
    print("NEG PRECISION@10: %2.5f" % neg_precision)
    print("NEG HR@10: %2.5f" % neg_hr)
    print("################################")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    
    tuning_score = ((1/total_rmse)*neg_ndcg)**0.5
    return tuning_score, total_rmse, neg_ndcg
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
                
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--encoder", type=str2bool, default=False)
    parser.add_argument("--decoder", type=str2bool, default=True)
    parser.add_argument("--moe", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--eval", type = str2bool, default=False)
    parser.add_argument("--tune", type = str2bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    # dataset args
    parser.add_argument("--dataset", type = str, default="ciao_timestamp", help = "ciao, epinions")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="percentage of valid/test dataset")
    parser.add_argument('--user_seq_len', type=int, default=30, help="user random walk sequence length")
    parser.add_argument('--item_per_user', type=int, default=5, help="number of items per user")
    parser.add_argument('--augs', type=int, default=1, help="how many times augment train data per anchor user")
    parser.add_argument('--regen', type=str, default='no', help="[no, all, rw, total, train]")    
    parser.add_argument('--bs', type=int, default=32, help="Batch size of dataloader")
    parser.add_argument('--neg', type=str2bool, default=False)

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
        'enc_scheduler': args.enc_scheduler,
        'dec_scheduler': args.dec_scheduler,
    }
    
def run(config, checkpoint_dir=None):
    import torch, random, numpy as np # 함수 내부에서 import 해줘야 함
    
    os.chdir(Path(__file__).parent.resolve())
    if 'args_dict' in config:
        args = argparse.Namespace(**config['args_dict'])
    else:
        args = argparse.Namespace(**config)
    
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
    train_ds = DecoderDataset(total_train)
    valid_ds = DecoderDataset(total_valid)
    test_ds = DecoderDataset(total_test)
    
    model_config = Config[args.dataset]["model"].copy()
    training_config = Config[args.dataset]["training"].copy()   
    
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
    
    test_bs = 1024
    ds_iter = {
            "train_enc":DataLoader(train_enc, batch_size = training_config['bs_enc'], shuffle=True, num_workers=4),
            "train":DataLoader(train_ds, batch_size = training_config['bs_dec'], shuffle=True, num_workers=4),
            "valid":DataLoader(valid_ds, batch_size = test_bs, shuffle=False, num_workers=1),
            "test":DataLoader(test_ds, batch_size = test_bs, shuffle=False, num_workers=1)
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
        # device = torch.device(f'cuda:{args.id}' if torch.cuda.is_available() else 'cpu')
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    # print(f"GPU index: {device.index}")
    print(f"GPU : {device}")
    print("\n")
    pynvml.nvmlShutdown()
        
    ### model preparation ###
    model = Transformer(**model_config).to(device)
    if not args.eval:
        # Encoder
        opt_enc = torch.optim.AdamW(
            model.encoder.parameters(),
            lr = training_config['lr_enc'],
            betas=[0.9,0.999],
            weight_decay=training_config['weight_decay_enc'])
        
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
            best_model = train(device, model, opt_dec, lr_scheduler, ds_iter, training_config)
            model.load_state_dict(best_model)
            score, best_rmse, best_ndcg = eval(device, model, ds_iter)
            
            # 메모리 정리
            del model, opt_enc, opt_dec
            torch.cuda.empty_cache()
            gc.collect()

    else:
        ### eval ###
        checkpoint_path = training_config['checkpoint_path']
        print(checkpoint_path)
        if os.path.exists(checkpoint_path): #and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
            model.eval()
            with torch.no_grad():
                batch = next(iter(ds_iter['train']))
                batch = {k:v.to(device) for k,v in batch.items()}
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint["model_state_dict"])
                print("loading the best model from: " + checkpoint_path)
                score, best_rmse, best_ndcg = eval(device, model, ds_iter)
                _ = model(batch)
                torch.save(model.encoder.global_attention.cpu(), 'soft_attn.pt') # attention map 저장
        else:
            print("No Best Model Found")
            
        print(model_config)
        training_config = {k:v for k,v in training_config.items() if k not in ['checkpoint_path', 'enc_checkpoint_path']}
        print(training_config)
    
    if args.tune:
        if args.decoder:
            session.report({
                            'enc_rmse': enc_rmse.cpu().item(),
                            'score': score.cpu().item(),
                            'rmse': best_rmse.cpu().item(),
                            'ndcg': best_ndcg.item()
                            })
        else:
            session.report({
                            'enc_rmse': enc_rmse.cpu().item()
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
        search_space = {
            'args_dict': args_dict,
            # 'user_seq_len':tune.choice([20,30,40]),
            # 'item_per_user':tune.choice([2,3,4,5,6]),
            # 'enc_blocks': tune.choice([1,2,3]),
            # 'dec_blocks': tune.choice([1,2,3]),
            # 'n_experts': tune.choice([2,3,4,5,6,7,8]),
            # 'topk': tune.sample_from(lambda spec:random.randint(1,spec.config['n_experts']-1)),
            'num_heads': tune.choice(list(np.arange(5,9))),
            'd_model': tune.sample_from(lambda spec:random.choice([spec.config['num_heads']*64])),
            'd_ffn': tune.choice([256, 512]),
            # 'dropout': tune.choice([0.1, 0.2, 0.3]),
            'weight_decay_enc': tune.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.1]),
            'lr_enc': tune.choice([0.002, 0.003, 0.004, 0.005, 0.006]),
            # 'weight_decay_dec': tune.choice(list(np.arange(1e-1, 1e-2-1e-9, -0.01))+list(np.arange(1e-2, 1e-3-1e-9, -0.001))+list(np.arange(1e-3, 1e-4-1e-9, -0.0001))),
            # 'lr': tune.choice(list(np.arange(1e-1, 1e-2-1e-9, -0.01))+list(np.arange(1e-2, 1e-3-1e-9, -0.001))+list(np.arange(1e-3, 1e-4-1e-9, -0.0001))),
            # 'bs_enc':tune.choice([32,64,128]),
            'bs ':tune.choice([32,64,128,256])
        }
        
        # ray 초기화 및 실행
        if not ray.is_initialized():
            ray.init(
                num_cpus=24, num_gpus=4,
                ignore_reinit_error=True)
            
        if args.decoder:
            metric = 'score'
            mode = 'max'
        else:
            metric = 'enc_rmse'
            mode = 'min'
            
        part = 'full' if args.decoder else 'enc'
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d_%H%M")
        analysis = tune.run(run, config=search_space, num_samples=50,
                        resources_per_trial={'cpu':4, 'gpu':1},
                        raise_on_failed_trial=False, 
                        checkpoint_freq=0,
                        storage_path='/home/mlsys/workspace/BK/socialRecFormer_dummy/ray_results',
                        # name=f'{args.dataset}_{part}_{now_str}',
                        name=f'{args.dataset}_{part}_bs_128',
                        metric=metric, mode=mode
                        )
                    
        best_trial = analysis.get_best_trial(metric, mode=mode)
        print(f"\nBest trial: {best_trial.trial_id}")
        print(f"Best config: {best_trial.config}")
        print(f"Best score: {best_trial.last_result[metric]:.6f}")
        if args.decoder:
            print(f"Best RMSE: {best_trial.last_result['rmse']:.6f}")
            print(f"Best NDCG: {best_trial.last_result['ndcg']:.6f}")
        # used_config_file = os.path.join(best_trial.logdir, "used_config.json")

        sorted_trials = sorted(analysis.trials, 
                               key=lambda t:t.last_result.get(metric, float('inf')),
                               reverse=(mode=='max'))
        
        for t in sorted_trials:
            print(f"Trial: {t.trial_id}")
            print(f"{metric}: {t.last_result.get(metric)}")
            if metric=='score':
                print(f"NDCG: {t.last_result.get('rmse')}")
                print(f"RMSE: {t.last_result.get('ndcg')}")
            print(f"Config: {t.config}")
            print(f"Path: {t.local_path}")
            print("-" * 60)
    
    else:
        run(config=args_dict)

if __name__ == '__main__':
    main()