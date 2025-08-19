import os
import sys
import logging
import argparse
import random
import math
import json
import time
import itertools
import pynvml
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data_making_2 as dm
from utils import *
from config import Config
from dataset import MyDataset
from models.transformer import Transformer
from scheduler import WarmupCosineSchedule

# Ray Tune
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

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

def valid(model, ds_iter, epoch, checkpoint_path, global_step, best_rmse, best_mae, best_ndcg, update_cnt):
    eval_losses = AverageMeter()
    org_losses = AverageMeter()
    dec_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        # FIXME: valid를 기준으로 저장 X, test를 기준으로 바로 저장. 
        epoch_iterator = tqdm(ds_iter['valid'],
                              desc="Validating (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              leave=False)
        pred, trg, msk = [], [], []
        precision_5, recall_5, ndcg_5 = .0, .0, .0
        precision_10, recall_10, ndcg_10 = .0, .0, .0
        for step, batch in enumerate(epoch_iterator):
            batch['user_seq'] = batch['user_seq'].to(device)
            batch['user_degree'] = batch['user_degree'].to(device)
            batch['item_list'] = batch['item_list'].to(device)
            batch['item_degree'] = batch['item_degree'].to(device)
            batch['item_rating'] = batch['item_rating'].to(device)
            # batch['spd_matrix'] = batch['spd_matrix'].to(device)
            ##################### [DEV] #####################
            batch['anchor_user'] = batch['anchor_user'].to(device)
            batch['anchor_degree'] = batch['anchor_degree'].to(device)
            batch['anchor_items'] = batch['anchor_items'].to(device)
            batch['anchor_item_degree'] = batch['anchor_item_degree'].to(device)
            batch['imp_fdback'] = batch['imp_fdback'].to(device)
            batch['exp_fdback'] = batch['exp_fdback'].to(device)

            rank_logits, rating_pred, enc_loss, dec_rmse = model(batch, is_train=False)
            # rmse loss 계산(Rating)
            mask = (batch['item_rating'] != 0)
            org_loss = RMSE(rating_pred, batch['item_rating'], mask)
            loss = org_loss
            
            eval_losses.update(loss)
            org_losses.update(org_loss)
            
            pred.append(rating_pred)
            trg.append(batch['item_rating'])
            msk.append(mask)

            # Rank Valid Result
            rank_eval_10 = RankMetric(batch['anchor_items'], batch['imp_fdback'], batch['exp_fdback'], rank_logits, k=10)
            precision_10 += rank_eval_10.precision()
            recall_10 += rank_eval_10.recall()
            ndcg_10 += rank_eval_10.NDCG()

            epoch_iterator.set_description(
                        "Validating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), loss))
        
        # Rating Valid Result
        pred = torch.cat(pred)
        trg = torch.cat(trg)
        msk = torch.cat(msk)
        
        total_rmse = RMSE(pred, trg, msk)
        total_mae = MAE(pred, trg, msk)

        precision_10 /= (step+1)
        recall_10 /= (step+1)
        ndcg_10 /= (step+1)
        
        print(f"Precision : {precision_10:.4f} / Recall : {recall_10:.4f} / NDCG : {ndcg_10:.4f}")
        
        if ((1/total_rmse)*0.4+(ndcg_10)*0.6 > (1/best_rmse)*0.4+(best_ndcg)*0.6): 
            best_ndcg = ndcg_10
            best_rmse = total_rmse
            best_mae = total_mae
            torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
            print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}, test NDCG@10 = {ndcg_10.item():.6f}')
            update_cnt = 0
        
        else:
            update_cnt += 1

    return eval_losses.avg, best_rmse, best_mae, best_ndcg, total_rmse, total_mae, update_cnt, org_losses.avg, dec_losses.avg

def train(model, optimizer, lr_scheduler, ds_iter, training_config, writer):
    global baseline_rmse, baseline_mae

    # TODO: Epoch당 loss, RMSE, MAE 추적 => TensorBoard 또는 파일 저장을 통해 tracing할 수 있도록.
    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", training_config["num_train_steps"])

    checkpoint_path = training_config['checkpoint_path']
    best_rmse = 9999.0
    best_mae = 9999.0
    best_ndcg = 0
    baseline_rmse = training_config['baseline_rmse']
    baseline_mae = training_config['baseline_mae']

    total_epochs = training_config["num_epochs"]

    model.train()
    init_t = time.time()
    total_time = 0
    update_cnt = 0
    
    if device.type=='cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device=device)
        start.record(stream)

    # Training step
    for epoch in range(total_epochs):
        losses = AverageMeter()
        org_losses = AverageMeter()
        epoch_iterator = tqdm(ds_iter['train'],
                            desc="Training (X / X Steps) (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True,
                            leave=False)
        
        for step, batch in enumerate(epoch_iterator):
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].to(device)
            batch['user_degree'] = batch['user_degree'].to(device)
            batch['item_list'] = batch['item_list'].to(device)
            batch['item_degree'] = batch['item_degree'].to(device)
            batch['item_rating'] = batch['item_rating'].to(device)
            # batch['spd_matrix'] = batch['spd_matrix'].to(device)
            ##################### [DEV] #####################
            batch['anchor_user'] = batch['anchor_user'].to(device)
            batch['anchor_degree'] = batch['anchor_degree'].to(device)
            batch['anchor_items'] = batch['anchor_items'].to(device)
            batch['anchor_item_degree'] = batch['anchor_item_degree'].to(device)
            batch['imp_fdback'] = batch['imp_fdback'].to(device)
            batch['exp_fdback'] = batch['exp_fdback'].to(device)

            # forward pass
            rank_logits, rating_pred, enc_loss, dec_rmse = model(batch)

            # compute loss
            mask = (batch['item_rating'] != 0)
            org_loss = MSE(rating_pred, batch['item_rating'], mask)
            y_rank_value = F.softmax(batch['exp_fdback'].float(), dim=-1)
            rank_loss = MSE(rank_logits, y_rank_value)
            loss = org_loss + rank_loss
            # loss = org_loss
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss)
            org_losses.update(org_loss)
            epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), losses.val))
            
        # validation
        if device.type=='cuda':
            end.record(stream)
            torch.cuda.synchronize()
            
        total_time += (start.elapsed_time(end))
        valid_loss, best_rmse, best_mae, best_ndcg, valid_rmse, valid_mae, update_cnt, org_loss, dec_loss = valid(model, ds_iter, epoch, checkpoint_path, step, best_rmse, best_mae, best_ndcg, update_cnt)
        lr_scheduler.step(valid_loss) # ReduceLROnPlateau

        # Tensorboard recording
        writer.add_scalars('Loss', {'Train':losses.avg, 'Valid':valid_loss,}, epoch)
        writer.add_scalar('RMSE/Test', valid_rmse, epoch)
        writer.add_scalar('MAE/Test', valid_mae, epoch)

        # # Ray recording
        # tune.report({"loss":valid_loss})

        print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || epoch MAE: {valid_mae:.4f} || best RMSE: {best_rmse:.4f} || best MAE: {best_mae:.4f} || best NDCG@10: {best_ndcg:.4f}\n")
        if epoch > 100:
            break
        if update_cnt > 15: 
            break
    writer.close()

    print('\n [Train Finished]')
    print("total training time (s): {}".format((time.time()-init_t)))
    print("total training time (ms): {}".format(total_time))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("total memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=device.index))


def eval(model, ds_iter):

    eval_losses = AverageMeter()
    model.eval()

    if device.type=='cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device=device)
        start.record(stream)
        
    with torch.no_grad():
        epoch_iterator = tqdm(ds_iter['test'],
                        desc="Validating (X / X Steps) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True,
                        leave=False)

        pred, trg, msk = [], [], []
        total_precision_5, total_precision_10 = 0.0, 0.0
        total_recall_5, total_recall_10 = 0.0, 0.0
        total_ndcg_5, total_ndcg_10 = 0.0, 0.0
        
        # NDCG : user별 중복 계산(input이 다르므로, 다른 결과 발생) -> 1. 그 중에서 best를 선택하는 코드
        ndcg_dict = dict()
        
        for step, batch in enumerate(epoch_iterator):
            
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].to(device)
            batch['user_degree'] = batch['user_degree'].to(device)
            batch['item_list'] = batch['item_list'].to(device)
            batch['item_degree'] = batch['item_degree'].to(device)
            batch['item_rating'] = batch['item_rating'].to(device)
            # batch['spd_matrix'] = batch['spd_matrix'].to(device)
            ##################### [DEV] #####################
            batch['anchor_user'] = batch['anchor_user'].to(device)
            batch['anchor_degree'] = batch['anchor_degree'].to(device)
            batch['anchor_items'] = batch['anchor_items'].to(device)
            batch['anchor_item_degree'] = batch['anchor_item_degree'].to(device)
            batch['imp_fdback'] = batch['imp_fdback'].to(device)
            batch['exp_fdback'] = batch['exp_fdback'].to(device)
            
            rank_logits, rating_pred, enc_loss, dec_rmse = model(batch, is_train=False)
            mask = (batch['item_rating'] != 0)
            
            loss = RMSE(rating_pred, batch['item_rating'], mask)
            eval_losses.update(loss)
            
            pred.append(rating_pred)
            trg.append(batch['item_rating'])
            msk.append(mask)
            
            # Rank Valid Result
            rank_eval_5 = RankMetric(batch['anchor_items'], batch['imp_fdback'], batch['exp_fdback'], rank_logits, k=5)
            precision_5 = rank_eval_5.precision()
            recall_5 = rank_eval_5.recall()
            ndcg_5 = rank_eval_5.NDCG()
            
            total_precision_5 += precision_5
            total_recall_5 += recall_5
            total_ndcg_5 += ndcg_5
            
            
            rank_eval_10 = RankMetric(batch['anchor_items'], batch['imp_fdback'], batch['exp_fdback'], rank_logits, k=10)
            precision_10 = rank_eval_10.precision()
            recall_10 = rank_eval_10.recall()
            ndcg_10 = rank_eval_10.NDCG()
            
            ndcg2_10 = rank_eval_10.NDCG2()
            for u,n in zip(batch['anchor_user'], ndcg2_10.squeeze()):
                if n.item() > ndcg_dict.get(u.item(),0):
                    ndcg_dict[u.item()] = n.item()
                else:
                    ndcg_dict[u.item()] = ndcg_dict.get(u,0)
            
            total_precision_10 += precision_10
            total_recall_10 += recall_10
            total_ndcg_10 += ndcg_10

            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), eval_losses.val))
        pred = torch.cat(pred)
        trg = torch.cat(trg)
        msk = torch.cat(msk)
        print(pred[msk])
        print(trg[msk])
        total_rmse = RMSE(pred, trg, msk)
        total_mae = MAE(pred, trg, msk)
        
        total_ndcg_5 /= (step+1)
        total_recall_5 /= (step+1)
        total_precision_5 /= (step+1)
        total_ndcg_10 /= (step+1)
        total_recall_10 /= (step+1)
        total_precision_10 /= (step+1)
        
        print(len(ndcg_dict.values()))
        total_ndcg2_10 = np.mean(list(ndcg_dict.values()))
        print(f"ndcg2@10 : {total_ndcg2_10}")

    if device.type=='cuda':
        end.record(stream)
        torch.cuda.synchronize()

    print("\n [Evaluation Results]")
    print("Loss: %2.5f" % eval_losses.avg)
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print(f"Precision@5 : {total_precision_5} / Recall@5 : {total_recall_5} / NDCG@5 : {total_ndcg_5}")
    print(f"Precision@10 : {total_precision_10} / Recall@10 : {total_recall_10} / NDCG@10 : {total_ndcg_10}")
    print(f"total eval time: {(start.elapsed_time(end))}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    
def eval2(model, ds_iter):
    model.eval()
    if device.type=='cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device=device)
        start.record(stream)
        
    with torch.no_grad():
        epoch_iterator = tqdm(ds_iter['test'],
                        desc="Validating (X / X Steps) (loss=X.X)",
                        ascii=" =",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True,
                        leave=False)

        total_rmse, total_mae = 0.0, 0.0
        total_precision_5, total_precision_10 = 0.0, 0.0
        total_recall_5, total_recall_10 = 0.0, 0.0
        total_ndcg_5, total_ndcg_10 = 0.0, 0.0
        
        # NDCG : user별 중복 계산(input이 다르므로, 다른 결과 발생) -> 1. 그 중에서 best를 선택하는 코드
        ndcg_df = pd.DataFrame()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator):     
                batch['user_seq'] = batch['user_seq'].to(device)
                batch['user_degree'] = batch['user_degree'].to(device)
                batch['item_list'] = batch['item_list'].to(device)
                batch['item_degree'] = batch['item_degree'].to(device)
                batch['item_rating'] = batch['item_rating'].to(device)
                batch['anchor_user'] = batch['anchor_user'].to(device)
                batch['anchor_degree'] = batch['anchor_degree'].to(device)
                batch['anchor_items'] = batch['anchor_items'].to(device)
                batch['anchor_item_degree'] = batch['anchor_item_degree'].to(device)
                batch['imp_fdback'] = batch['imp_fdback'].to(device)
                batch['exp_fdback'] = batch['exp_fdback'].to(device)
                
                rank_logits, rating_pred, _, _ = model(batch, is_train=False)
                mask = (batch['item_rating'] != 0)
                rmse = RMSE(rating_pred, batch['item_rating'], mask).item()
                mae = MAE(rating_pred, batch['item_rating'], mask)
                total_rmse += rmse
                total_mae += mae
                
                # rank metric                
                df = pd.DataFrame({'users':batch['anchor_user'].data.cpu().tolist(),
                                   'items':batch['anchor_items'].data.cpu().tolist(),
                                   'ratings':batch['exp_fdback'].data.cpu().tolist(),
                                   'logits':rank_logits.data.cpu().tolist()})
                ndcg_df = pd.concat([ndcg_df, df], axis=0)                
            
                epoch_iterator.set_description(
                            "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), rmse))
                
        # calculate NDCG
        items = torch.from_numpy(np.stack(ndcg_df['items'].values))
        ratings = torch.from_numpy(np.stack(ndcg_df['ratings'].values))
        logits = torch.from_numpy(np.stack(ndcg_df['logits'].values))
        new_k = []
        ndcg = []
        for i in range(items.size(0)):
            k = min((items[i]!=0).sum().item(),10)
            _,ideal_idx = torch.topk(ratings[i],k)
            ideal_items = torch.gather(items[i], -1, ideal_idx)
            ideal_ratings = torch.gather(ratings[i],-1,ideal_idx)
            # recommended topk
            _,rec_idx = torch.topk(logits[i], k)
            rec_items  = torch.gather(items[i],-1,rec_idx)
            rec_ratings = torch.gather(ratings[i],-1,rec_idx)
            # mask
            rowA = rec_items.unsqueeze(1)
            rowB = ideal_items.unsqueeze(0)
            mask = (rowA==rowB).any(dim=1)
            rec_ratings *= mask
            # dcg/idcg/ndcg
            discount = torch.log2(torch.arange(k)+2)
            dcg = torch.sum(rec_ratings/discount, dim=-1)
            idcg = torch.sum(ideal_ratings/discount, dim=-1)
            ndcg.append((dcg/(idcg+1e-10)).item())
            new_k.append(k)
        
        ndcg_df['NDCG'] = ndcg
        ndcg_df['new_k'] = new_k
        ndcg_mean = ndcg_df.groupby('users')['NDCG'].mean()
        total_ndcg = np.mean(ndcg_mean.values)
        print(total_ndcg)
        ndcg_df.to_csv(f'./ndcg_test_{args.dataset}.csv', index=False)
                
        total_rmse /= (step+1)
        total_mae /= (step+1)

    if device.type=='cuda':
        end.record(stream)
        torch.cuda.synchronize()

    print("\n [Evaluation Results]")
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print(f"Precision@5 : {total_precision_5} / Recall@5 : {total_recall_5} / NDCG@5 : {total_ndcg_5}")
    print(f"Precision@10 : {total_precision_10} / Recall@10 : {total_recall_10} / NDCG@10 : {total_ndcg_10}")
    print(f"total eval time: {(start.elapsed_time(end))}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    
def get_args():
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--eval", type = bool, default=False,
                        help="train eval")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, help="checkpoint model name")
    parser.add_argument('--num_layers_enc', type=int, default=3, help="num enc layers")
    parser.add_argument('--num_layers_dec', type=int, default=5, help="num dec layers")
    parser.add_argument('--n_experts', type=int, default=8, help="MoE number of total experts")
    parser.add_argument('--topk', type=int, default=2, help="MoE number of experts")
    parser.add_argument('--rating_thres', type=int, default=4, help="explicit rating threshold for creating implicit feedback")
    parser.add_argument('--lr', type=float, default=1e-3) # rating 기준 rw 생성의 경우 default = 1e-3
    # dataset args
    parser.add_argument("--dataset", type = str, default="epinions", help = "ciao, epinions")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="percentage of valid/test dataset")
    parser.add_argument('--user_seq_len', type=int, default=30, help="user random walk sequence length")
    parser.add_argument('--item_per_user', type=int, default=5, help="number of items per user")
    parser.add_argument('--return_params', type=int, default=1, help="return param value for generating random sequence")
    parser.add_argument('--train_augs', type=int, default=1, help="how many times augment train data per anchor user")    
    parser.add_argument('--test_augs', type=bool, default=True, help="Whether augment test data set in proportion to train_augs or not / max = 3")    
    parser.add_argument('--regenerate', type=bool, default=False, help="Whether regenerate dataframe(random walk & total df) or not")    
    parser.add_argument('--bs', type=int, default=128, help="Batch size of dataloader")
    
    args = parser.parse_args()
    return args

def main():
    global device, args

    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    ######################################################### data preparation #########################################################

    ### FIXME: 전체 데이터에 대해 파일 생성이 오래 걸림 (현재 시퀀스의 rating matrix 생성하는 부분이 문제로 보임)
        ### FIXME: (231012) validation set을 통해 모델이 잘 train 되는것은 확인했으므로, 바로 test를 진행하면서 model을 저장.


    # regenerate 여부 확인
    if args.regenerate:
        print("Re-Creating Datatset...")
    else:
        print("Loading Datatset...")
    data_making = dm.DatasetMaking(args)
    
    print("\n")
    total_train = data_making.total_train
    total_valid = data_making.total_valid
    total_test = data_making.total_test

    train_ds = MyDataset(total_train)
    valid_ds = MyDataset(total_valid)
    test_ds = MyDataset(total_test)


    ### get model config ###
    model_config = Config[args.dataset]["model"]
    training_config = Config[args.dataset]["training"]
    # batch size update
    training_config["batch_size"] = args.bs
    ds_iter = {
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            "valid":DataLoader(valid_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=4),
            "test":DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=4)
    }

    ######################################################### model initialization #########################################################

    training_config["learning_rate"] = args.lr
    # model config - num users & num items & degrees
    model_config["num_user"] = data_making.num_user
    model_config["num_item"] = data_making.num_item
    model_config["max_user_degree"] = data_making.max_user_degree
    model_config["max_item_degree"] = data_making.max_item_degree
    # model expansion (1) : Increase # of Encoder/Decoder Blocks
    model_config["num_layers_enc"] = args.num_layers_enc + int(math.log(args.train_augs,2))
    model_config["num_layers_dec"] = args.num_layers_dec + int(math.log(args.train_augs,2))
    
    # model expansion (2) : MoE topk router
    model_config["n_experts"] = args.n_experts
    # model expansion (2)-2 : MoE topk # of experts
    model_config["topk"] = args.topk + int(math.log(args.train_augs,2))
    
    # model expansion (3) : rating threshold for ranking task
    model_config["rating_thres"] = args.rating_thres

    ### log preparation ###
    log_dir = os.getcwd() + f'/logs/log_seed_{args.seed}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    ###  set the random seeds for deterministic results. ####
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    ### model preparation ###    # [batch_size, 1, len_k(=len_q)]
    print(model_config)
    model = Transformer(**model_config, args=args)

    # checkpoint_dir = os.getcwd() + f'/checkpoints/{args.dataset}/checkpoints_seed_{args.seed}/'
    checkpoint_data = os.getcwd() + f'/checkpoints/{args.dataset}/'
    if not os.path.exists(checkpoint_data):
        os.makedirs(checkpoint_data)
    checkpoint_dir = checkpoint_data + f'checkpoints_seed_{args.seed}/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, "train")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    name_dataset = str(args.dataset)
    name_seed = str(args.seed)
    name_u_len = str(args.user_seq_len)
    name_i_len = str(args.user_seq_len*args.item_per_user)
    name_n_enc = str(model_config['num_layers_enc'])
    name_n_dec = str(model_config['num_layers_dec'])
    name_train_augs = str(args.train_augs)
    name_test_augs = str(str(min(3,args.train_augs)) if args.test_augs else '')
    args.name = '_'.join([name_dataset, name_seed, name_u_len, name_i_len, name_n_enc, name_n_dec, name_train_augs, name_test_augs])
    checkpoint_path = os.path.join(checkpoint_dir, f'{args.name}.model') # set model name
    print(checkpoint_path, "\n")
    training_config["checkpoint_path"] = checkpoint_path

    # 1. file path check(train_augs & test_augs)
    train_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                              f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_train_{args.train_augs}times.pkl')
    if args.test_augs:
        print(f"dataset : {args.dataset}\n seed : {args.seed}\n test_ratio: {args.test_ratio}\n user_seq_len : {args.user_seq_len}\n item_seq_len : {name_i_len}\n return_params : {args.return_params}\n train_augs : {args.train_augs}\n test_augs : {args.train_augs}\n \
            num_enc_layers : {name_n_enc}\n num_dec_layers : {name_n_dec}")
        valid_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                              f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_valid_{args.train_augs}times.pkl')
        test_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                                f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_test_{args.train_augs}times.pkl')
    else:
        print(f"dataset : {args.dataset}\n seed : {args.seed}\n test_ratio: {args.test_ratio}\n user_seq_len : {args.user_seq_len}\n item_seq_len : {name_i_len}\n return_params : {args.return_params}\n train_augs : {args.train_augs}\n \
            num_enc_layers : {name_n_enc}\n num_dec_layers : {name_n_dec}")
        valid_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                              f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_valid.pkl')
        test_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                                f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_test.pkl')

    # gpu device선택
    device_ids = list(range(torch.cuda.device_count()))
    pynvml.nvmlInit()
    for i in device_ids:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        if util_info.gpu < 10:
            device = torch.device(f'cuda:{i}')
            break
        else:
            device = torch.device('cpu')
            

    pynvml.nvmlShutdown()

    if args.device=='cpu':
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"GPU index: {device.index}")
    print("\n")
    
    model = model.to(device)

    ############################################################ training preparation ############################################################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas=(0.9, 0.999), eps=1e-6, weight_decay=training_config["weight_decay"]
    )

    training_config["num_train_steps"] = len(ds_iter['train'])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        mode = 'min',
        factor = 0.85,
        patience = 3,
        threshold = 1e-2,
        # min_lr = 1e-6,
        verbose = True
    )

    ### TensorBoard writer preparation ###
    writer = SummaryWriter(os.path.join(log_dir,f"{args.name}.tensorboard"))
    ### train ###
    if not args.eval:
        train(model, optimizer, lr_scheduler, ds_iter, training_config, writer)

    # Since train logging is done by TensorBoard, log only test result.
    log_path = os.path.join(log_dir,'{}.log'.format(args.name))
    redirect_stdout(open(log_path, 'w'))

    ### eval ###
    print(checkpoint_path)
    if os.path.exists(checkpoint_path): #and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
        eval2(model, ds_iter)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()