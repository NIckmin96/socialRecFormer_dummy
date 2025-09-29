import os
import sys
import logging
import argparse
import random
import math
import pickle
import time
import itertools
import pynvml
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
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
        
# calculate NDCG(row별로 계산 : 알고리즘은 동일)
def NDCG(items, logits, ratings):
    items = torch.tensor(items)
    logits = torch.tensor(logits)
    ratings = torch.tensor(ratings)
    k = min((items!=0).sum().item(), 10)
    # ideal
    _,ideal_idx = torch.topk(ratings, k)
    ideal_items = torch.gather(items, -1, ideal_idx)
    ideal_ratings = torch.gather(ratings, -1, ideal_idx)
    # recommended
    _,rec_idx = torch.topk(logits, k)
    rec_items  = torch.gather(items,-1,rec_idx)
    rec_ratings = torch.gather(ratings,-1,rec_idx)
    # mask (ideal에 존재하는지 여부)
    rowA = rec_items.unsqueeze(1)
    rowB = ideal_items.unsqueeze(0)
    mask = (rowA==rowB).any(dim=1)
    rec_ratings *= mask
    # dcg/idcg/ndcg
    discount = torch.log2(torch.arange(k)+2)
    dcg = torch.sum(rec_ratings/discount, dim=-1)
    idcg = torch.sum(ideal_ratings/discount, dim=-1)
    ndcg = dcg/(idcg+1e-10).item()
    
    return ndcg

def BPR(output_batch, rating_batch, neg):
    bpr_loss = 0.0
    
    if neg:
        logits = output_batch[rating_batch==0]
        logit_0 = logits.sum() if logits.numel()==0 else logits.mean()
        
    logits = output_batch[rating_batch==1]
    logit_1 = logits.sum() if logits.numel()==0 else logits.mean()
    
    logits = output_batch[rating_batch==1]
    logit_1 = logits.sum() if logits.numel()==0 else logits.mean()
    
    logits = output_batch[rating_batch==2]
    logit_2 = logits.sum() if logits.numel()==0 else logits.mean()
    
    logits = output_batch[rating_batch==3]
    logit_3 = logits.sum() if logits.numel()==0 else logits.mean()
    
    logits = output_batch[rating_batch==4]
    logit_4 = logits.sum() if logits.numel()==0 else logits.mean()
    
    logits = output_batch[rating_batch==5]
    logit_5 = logits.sum() if logits.numel()==0 else logits.mean()
    
    if neg:
        for neg,pos in [(logit_0, logit_1), (logit_1, logit_2), (logit_2, logit_3), (logit_3, logit_4), (logit_4, logit_5)]:
            diff = pos-(neg+0.01)
            # loss = -F.logsigmoid(diff)
            loss = -F.log(diff if diff>=1 else -diff)
            bpr_loss += loss
    
    else:
        for neg,pos in [(logit_1, logit_2), (logit_2, logit_3), (logit_3, logit_4), (logit_4, logit_5)]:
            diff = pos-(neg+0.01)
            # loss = -F.logsigmoid(diff)
            loss = -F.log(diff if diff>=1 else -diff)
            bpr_loss += loss
        
    return bpr_loss

def train(model, optimizer, lr_scheduler, ds_iter, training_config, writer):

    # TODO: Epoch당 loss, RMSE, MAE 추적 => TensorBoard 또는 파일 저장을 통해 tracing할 수 있도록.
    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", len(ds_iter['train']))

    best_rmse = 9999.0
    best_mae = 9999.0
    best_ndcg = 0

    checkpoint_path = training_config['checkpoint_path']
    total_epochs = training_config["num_epochs"]

    model.train()
    init_t = time.time()
    total_time = 0
    update_cnt = 0
    
    # if device.type=='cuda':
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     stream = torch.cuda.current_stream(device=device)
    #     start.record(stream)

    
        # validation
        # if device.type=='cuda':
        #     end.record(stream)
        #     torch.cuda.synchronize()
        
    lr_lst = []
    metrics = Metrics()
    for epoch in range(total_epochs):
        losses = AverageMeter()
        main_losses = AverageMeter()
        sub_losses = AverageMeter()
        rank_losses = AverageMeter()
        # decoder 학습
        dec_iterator = tqdm(ds_iter['train'], desc="Decoder (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dec_iterator):
            batch = {k:v.to(device) for k,v in batch.items()}
            # forward pass
            global_preference, local_preference = model(batch)

            sub_mask = (batch['item_rating']!=0)
            sub_loss = metrics.RMSE(global_preference, batch['item_rating'], sub_mask)
            sub_losses.update(sub_loss.item())
            
            main_mask = (batch['anchor_ratings'] != 0)
            # print(main_mask.size(), local_preference.size(), batch['anchor_ratings'].size())
            main_loss = metrics.RMSE(local_preference, batch['anchor_ratings'], main_mask)
            main_losses.update(main_loss.item())

            rank_mask = (batch['anchor_items'] != 0)
            rank_logit = F.log_softmax(local_preference, dim=-1).float()
            rank_target = F.softmax(batch['anchor_ratings'], dim=-1)
            rank_loss = F.kl_div(rank_logit, rank_target, reduction='batchmean')
            # rank_loss = metrics.BPR(output, batch['anchor_ratings'].float(), args.neg) # 추후에, 하나로 합친 결과에 대한 loss계산하는 방식으로 추가 실험
            rank_losses.update(rank_loss.item())
            
    
            loss = main_loss + rank_loss + sub_loss
            losses.update(loss.item())
            
            
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()            
            dec_iterator.set_description(
                        "Decoder Training (%d / %d Steps) (loss=%2.5f)" % (step, len(dec_iterator), losses.avg))
            
        # total_time += (start.elapsed_time(end))
        # valid_loss, best_rmse, best_mae, valid_rmse, valid_mae, update_cnt = valid(model, ds_iter, epoch, checkpoint_path, step, best_rmse, best_mae, best_ndcg, update_cnt)
        valid_loss, best_rmse, best_mae, best_ndcg, valid_ndcg, valid_rmse, valid_mae, update_cnt = valid(model, ds_iter, epoch, checkpoint_path, step, best_rmse, best_mae, best_ndcg, update_cnt)
        if args.scheduler=='rp':
            lr_scheduler.step(valid_rmse)
        else:
            lr_scheduler.step()
            print(lr_scheduler.get_lr())
            lr_lst.extend(lr_scheduler.get_lr())
        

        # Tensorboard recording
        writer.add_scalars('Loss', {'Train':losses.avg, 'Valid':valid_loss,}, epoch)
        writer.add_scalar('RMSE/Test', valid_rmse, epoch)
        writer.add_scalar('MAE/Test', valid_mae, epoch)

        # epoch_rank_loss = rank_losses.avg

        print(f"Epoch {epoch:03d}: Main Loss: {main_losses.avg:.4f} || Sub Loss: {sub_losses.avg:.4f} || Rank Loss: {rank_losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch NDCG@10: {valid_ndcg:.4f} || epoch RMSE: {valid_rmse:.4f} || best RMSE: {best_rmse:.4f} || best NDCG@10: {best_ndcg:.4f} ||\n")
        # print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch NDCG@10: {valid_ndcg:.4f} || epoch RMSE: {valid_rmse:.4f} || epoch MAE: {valid_mae:.4f} || best RMSE: {best_rmse:.4f} || best MAE: {best_mae:.4f} || best NDCG@10: {best_ndcg:.4f} ||\n")
        # print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || epoch MAE: {valid_mae:.4f} || best RMSE: {best_rmse:.4f} || best MAE: {best_mae:.4f} ||\n")
        if update_cnt > 20: 
            break
    writer.close()

    print('\n [Train Finished]')
    print("total training time (s): {}".format((time.time()-init_t)))
    # print("total training time (ms): {}".format(total_time))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("total memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=device.index))
    # lr 저장
    with open('lr_lst.pkl', 'wb') as f :
        pickle.dump(lr_lst, f)
        

def valid(model, ds_iter, epoch, checkpoint_path, global_step, best_rmse, best_mae, best_ndcg, update_cnt):
    eval_losses = AverageMeter()
    model.eval()
    
    metrics = Metrics()
    total_rmse, total_mae = 0.0, 0.0
    output_df = pd.DataFrame()
    with torch.no_grad():
        
        dec_iterator = tqdm(ds_iter['valid'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(dec_iterator):     
            batch = {k:v.to(device) for k,v in batch.items()}
            
            global_preference, local_preference = model(batch)
            
            # mask = (batch['item_rating'] != 0)
            # rmse = metrics.RMSE(global_preference, batch['item_rating'], mask).item()
            # mae = metrics.MAE(global_preference, batch['item_rating'], mask).item()
            
            mask = (batch['anchor_ratings'] != 0)
            rmse = metrics.RMSE(local_preference, batch['anchor_ratings'], mask).item()
            mae = metrics.MAE(local_preference, batch['anchor_ratings'], mask).item()
            
            total_rmse += rmse
            total_mae += mae
            eval_losses.update(rmse)
            
            dec_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(dec_iterator), rmse))
            
            # zero padding인 경우 제외
            mask = (batch['anchor_items']!=0)
            local_preference = local_preference*mask
            
            item_lst, rating_lst, output_lst = [],[],[]
            for i in range(batch['anchor_user'].size(0)):
                mask = (batch['anchor_items'][i]!=0) # padding되지 않은 index
                items = items = batch['anchor_items'][i][mask].data.cpu().tolist()
                ratings = batch['anchor_ratings'][i][mask].data.cpu().tolist()
                outputs = local_preference[i][mask].data.cpu().tolist()
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
            
    output_df = output_df.groupby('anchor_user').agg({'anchor_items':lambda x:sum(x, start=[]),
                                                          'anchor_ratings':lambda x:sum(x, start=[]),
                                                          'outputs':lambda x:sum(x, start=[])}).reset_index()
    output_df['logits'] = output_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1))
    output_df['targets'] = output_df['anchor_ratings'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1))    
    output_df['ndcg'] = output_df.apply(lambda x:metrics.NDCG(x['anchor_items'], x['logits'], x['anchor_ratings'], 10), axis=1)
    output_df = output_df.dropna(how='any')
    total_ndcg = output_df[output_df['anchor_items'].apply(len)>=10]['ndcg'].mean()
    total_rmse /= (step+1)
    total_mae /= (step+1)
            
    if ((1/total_rmse)*0.5+(total_ndcg)*0.5 > (1/best_rmse)*0.5+(best_ndcg)*0.5): 
        best_ndcg = total_ndcg
        best_rmse = total_rmse
        best_mae = total_mae
        torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
        print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse:.6f}, test MAE = {total_mae:.6f}, test NDCG@10 = {best_ndcg:.6f}')
        update_cnt = 0
    elif total_rmse<=best_rmse:
        pass
    else:
        update_cnt += 1

    return eval_losses.avg, best_rmse, best_mae, best_ndcg, total_ndcg, total_rmse, total_mae, update_cnt
    # return eval_losses.avg, best_rmse, best_mae, total_rmse, total_mae, update_cnt
    
def eval(model, ds_iter):
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
    output_df = pd.DataFrame()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator):     
            batch = {k:v.to(device) for k,v in batch.items()}
            
            global_preference, local_preference = model(batch)
            
            mask = (batch['anchor_ratings'] != 0)
            rmse = metrics.RMSE(local_preference, batch['anchor_ratings'], mask).item()
            mae = metrics.MAE(local_preference, batch['anchor_ratings'], mask)
            total_rmse += rmse
            total_mae += mae
            
            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), rmse))
            
            # zero padding인 경우 제외
            mask = (batch['anchor_items']!=0)
            # output = output[mask]
            
            item_lst, rating_lst, output_lst = [],[],[]
            for i in range(batch['anchor_user'].size(0)): # batch내에서 user별 iteration
                mask = (batch['anchor_items'][i]!=0) # padding되지 않은 index
                items = batch['anchor_items'][i][mask].data.cpu().tolist()
                ratings = batch['anchor_ratings'][i][mask].data.cpu().tolist()
                outputs = local_preference[i][mask].data.cpu().tolist()
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
    neg_recall = neg_['recall@10'].mean()
    # negative sampling 미포함 + item 개수별 filter x
    non_neg = non_neg_df.apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    non_neg = non_neg.dropna(how='any')
    non_neg_ndcg = non_neg['ndcg@10'].mean()
    non_neg_precision = non_neg['precision@10'].mean()
    non_neg_recall = non_neg['recall@10'].mean()
    # non neg + item 개수별 filter o
    filtered = non_neg_df[non_neg_df['anchor_items'].apply(len)>=10].apply(lambda x:metrics.rank_metrics(x, 10), axis=1)
    filtered_ndcg = filtered['ndcg@10'].mean()
    filtered_precision = filtered['precision@10'].mean()
    filtered_recall = filtered['recall@10'].mean()
    total_rmse /= (step+1)
    total_mae /= (step+1)           
    
    neg_.to_csv(f'eval_output_{args.dataset}_{args.item_per_user}_neg.csv', index=False)
    non_neg.to_csv(f'eval_output_{args.dataset}_{args.item_per_user}_non_neg.csv', index=False)
        
    # if device.type=='cuda':
    #     end.record(stream)
    #     torch.cuda.synchronize()

    print("\n [Evaluation Results]")
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print("################################")
    print("NEG NDCG@10: %2.5f" % neg_ndcg)
    print("NEG RECALL@10: %2.5f" % neg_recall)
    print("NEG PRECISION@10: %2.5f" % neg_precision)
    print("################################")
    print("TOTAL NDCG@10: %2.5f" % non_neg_ndcg)
    print("TOTAL RECALL@10: %2.5f" % non_neg_recall)
    print("TOTAL PRECISION@10: %2.5f" % non_neg_precision)
    print("################################")
    print("FILTERED NDCG@10: %2.5f" % filtered_ndcg)
    print("FILTERED RECALL@10: %2.5f" % filtered_recall)
    print("FILTERED PRECISION@10: %2.5f" % filtered_precision)
    # print(f"Precision@5 : {total_precision_5} / Recall@5 : {total_recall_5} / NDCG@5 : {total_ndcg_5}")
    # print(f"Precision@10 : {total_precision_10} / Recall@10 : {total_recall_10} / NDCG@10 : {total_ndcg_10}")
    # print(f"total eval time: {(start.elapsed_time(end))}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    
def get_args():
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--encoder", type=bool, default=False)
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--eval", type = bool, default=False)
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, help="checkpoint model name")
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=2, help="num enc layers")
    parser.add_argument('--enc_blocks', type=int, default=1, help="num enc layers")
    parser.add_argument('--dec_blocks', type=int, default=1, help="num dec layers")
    parser.add_argument('--dropout', type=float, default=0.1, help="num dec layers")
    parser.add_argument('--n_experts', type=int, default=4, help="MoE number of total experts")
    parser.add_argument('--topk', type=int, default=3, help="MoE number of experts")
    parser.add_argument('--lr_enc', type=float, default=8e-3)
    parser.add_argument('--lr', type=float, default=5e-3) 
    # dataset args
    parser.add_argument("--dataset", type = str, default="ciao_timestamp", help = "ciao, epinions")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="percentage of valid/test dataset")
    parser.add_argument('--user_seq_len', type=int, default=30, help="user random walk sequence length")
    parser.add_argument('--item_per_user', type=int, default=5, help="number of items per user")
    parser.add_argument('--augs', type=int, default=1, help="how many times augment train data per anchor user")
    parser.add_argument('--regen', type=str, default='no', help="[no, all, rw, total, train]")    
    parser.add_argument('--bs', type=int, default=128, help="Batch size of dataloader")
    parser.add_argument('--neg', type=bool, default=False)

    # tuning
    parser.add_argument('--scheduler', type=str, default='cs')
    
    args = parser.parse_args()
    return args

def main():
    global device, args

    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    ######################################################### data preparation #########################################################


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
    # encoder
    total_train = data_making.total_train
    total_valid = data_making.total_valid
    total_test = data_making.total_test
    
    min_item_len = data_making.min_item_len

    ### get model config ###
    model_config = Config[args.dataset]["model"]
    training_config = Config[args.dataset]["training"]
    # batch size update
    training_config["batch_size"] = args.bs
    
    # dataset & dataloader
    train_enc = EncoderDataset(total_train)
    train_ds = DecoderDataset(total_train)
    valid_ds = DecoderDataset(total_valid)
    test_ds = DecoderDataset(total_test)
    
    ds_iter = {
            "train_enc":DataLoader(train_enc, batch_size = training_config["batch_size"], shuffle=True, num_workers=1),
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=1),
            "valid":DataLoader(valid_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=1),
            "test":DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=1)
    }
    
    ######################################################### model initialization #########################################################
    
    # model config - num users & num items & degrees
    model_config["user_seq_len"] = args.user_seq_len
    model_config["item_seq_len"] = args.user_seq_len*args.item_per_user
    model_config['min_item_len'] = min_item_len
    model_config["num_user"] = data_making.num_user
    model_config["num_item"] = data_making.num_item
    model_config["max_user_degree"] = data_making.max_user_degree
    model_config["max_item_degree"] = data_making.max_item_degree
    # model expansion (1) : Increase # of Encoder/Decoder Blocks
    model_config["num_heads"] = args.num_heads
    model_config["enc_blocks"] = args.enc_blocks
    model_config["dec_blocks"] = args.dec_blocks
    model_config["d_model"] = args.d_model
    model_config["d_ffn"] = args.d_ffn
    model_config["dropout"] = args.dropout
    
    # model expansion (2) : MoE topk router
    model_config["n_experts"] = args.n_experts
    # model expansion (2)-2 : MoE topk # of experts
    model_config["topk"] = args.topk

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
    model = Transformer(**model_config)

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
    
    name_seed = str(args.seed)
    name_u_len = str(args.user_seq_len)
    name_i_len = str(args.user_seq_len*args.item_per_user)
    name_augs = str(args.augs)
    name_n_heads = str(args.num_heads)
    name_n_enc = str(args.enc_blocks)
    name_n_dec = str(args.dec_blocks)
    name_d_model = str(model_config['d_model'])
    name_d_ffn = str(model_config['d_ffn'])
    name_lr = str(args.lr)
    name_lr_enc = str(args.lr_enc)
    name = '_'.join([name_seed, name_u_len, name_i_len, name_augs, name_n_heads, name_n_dec, name_d_model, name_d_ffn, name_lr, args.scheduler])
    # enc_name = '_'.join([name_seed, name_u_len, name_i_len, name_augs, name_n_heads, name_n_enc, name_d_model, name_d_ffn, name_lr_enc, args.enc_scheduler])
    checkpoint_path = os.path.join(checkpoint_dir, f'{name}_enc_only.model') # set model name
    # checkpoint_enc = os.path.join(checkpoint_dir, f'{enc_name}_enc.model') # set model name
    print(checkpoint_path, "\n")
    training_config["checkpoint_path"] = checkpoint_path

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
        device = torch.device(f'cuda:{args.id}' if torch.cuda.is_available() else 'cpu')
    
    print(f"GPU index: {device.index}")
    print("\n")
    
    model = model.to(device)

    ############################################################ training preparation ############################################################   
    
    ### TensorBoard writer preparation ###
    writer = SummaryWriter(os.path.join(log_dir,f"{args.name}.tensorboard"))
    ### train ###
    if not args.eval:        
        
        optimizer = torch.optim.AdamW(
            # model.decoder.parameters(),
            model.parameters(),
            lr = args.lr,
            betas=[0.9,0.999],
            weight_decay=training_config['weight_decay'])
                    
        if args.scheduler=='rp':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            mode = 'min',
            factor = 0.9,
            patience = 2,
            min_lr=1e-5,
            threshold = 1e-3,
            verbose = True
            )
        else:
            lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=100,
            cycle_mult=1,
            max_lr = args.lr,
            min_lr=5e-5,
            warmup_steps=10,
            gamma=0.5,
            )
        
        train(model, optimizer, lr_scheduler, ds_iter, training_config, writer)

    # Since train logging is done by TensorBoard, log only test result.
    log_path = os.path.join(log_dir,'{}.log'.format(name))
    redirect_stdout(open(log_path, 'w'))

    ### eval ###
    print(checkpoint_path)
    if os.path.exists(checkpoint_path): #and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
        eval(model, ds_iter)
        ############################################################################
        with torch.no_grad():
            batch = next(iter(ds_iter['train']))
            batch = {k:v.to(device) for k,v in batch.items()}
            _, _ = model(batch)
            torch.save(model.encoder.global_attention.cpu(), 'global_attn_abl2.pt') # attention map 저장     
        ############################################################################

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()