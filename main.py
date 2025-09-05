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
from dataset import MyDataset
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

def BPR(output_batch, rating_batch):
    bpr_loss = 0.0
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
    
    for neg,pos in [(logit_1, logit_2), (logit_2, logit_3), (logit_3, logit_4), (logit_4, logit_5)]:
        diff = pos-(neg+0.1)
        loss = -F.logsigmoid(diff)
        bpr_loss += loss
        
    return bpr_loss
        

def valid(model, ds_iter, epoch, checkpoint_path, global_step, best_rmse, best_mae, best_ndcg, update_cnt):
    eval_losses = AverageMeter()
    model.eval()
    
    metrics = Metrics()
    total_rmse, total_mae = 0.0, 0.0
    output_df = pd.DataFrame()
    with torch.no_grad():
        epoch_iterator = tqdm(ds_iter['valid'], desc="Validating (X / X Steps) (loss=X.X)", bar_format="{l_bar}{r_bar}", dynamic_ncols=True, leave=False)
        for step, batch in enumerate(epoch_iterator):     
            batch = {k:v.to(device) for k,v in batch.items()}
            
            rank_output, rating_pred, _, _ = model(batch, is_train=False)
            mask = (batch['item_rating'] != 0)
            rmse = metrics.RMSE(rating_pred, batch['item_rating'], mask).item()
            # y_rank_value = F.softmax(batch['anchor_ratings'].float(), dim=-1)
            # rmse += RMSE(rank_output, y_rank_value)
            mae = metrics.MAE(rating_pred, batch['item_rating'], mask)
            
            total_rmse += rmse
            total_mae += mae
            eval_losses.update(rmse)
            
            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), rmse))
            
            # zero padding인 경우 제외
            mask = (batch['anchor_items']!=0)
            rank_output = rank_output*mask
            
            item_lst, rating_lst, output_lst = [],[],[]
            for i in range(batch['anchor_user'].size(0)):
                items = batch['anchor_items'][i][batch['anchor_items'][i]!=0].data.cpu().tolist()
                ratings = batch['anchor_ratings'][i][batch['anchor_ratings'][i]!=0].data.cpu().tolist()
                outputs = rank_output[i][rank_output[i]!=0].data.cpu().tolist()
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
    output_df['ndcg'] = output_df.apply(lambda x:metrics.NDCG(x['anchor_items'], x['logits'], x['anchor_ratings']), axis=1)
    output_df = output_df.dropna(how='any')
    total_ndcg = output_df[output_df['anchor_items'].apply(len)>=10]['ndcg'].mean()
    total_rmse /= (step+1)
    total_mae /= (step+1)
            
    if ((1/total_rmse)*0.1+(total_ndcg)*0.9 > (1/best_rmse)*0.1+(best_ndcg)*0.9): 
    # if (total_rmse < best_rmse) | (total_ndcg > best_ndcg): 
        best_ndcg = total_ndcg
        best_rmse = total_rmse
        best_mae = total_mae
        torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
        print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse:.6f}, test MAE = {total_mae:.6f}, test NDCG@10 = {best_ndcg:.6f}')
        update_cnt = 0
    
    else:
        update_cnt += 1

    return eval_losses.avg, best_rmse, best_mae, best_ndcg, total_ndcg, total_rmse, total_mae, update_cnt

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

    metrics = Metrics()
    # Training step
    for epoch in range(total_epochs):
        losses = AverageMeter()
        org_losses = AverageMeter()
        rank_losses = AverageMeter()
        epoch_iterator = tqdm(ds_iter['train'],
                            desc="Training (X / X Steps) (loss=X.X)",
                            bar_format="{l_bar}{r_bar}",
                            dynamic_ncols=True,
                            leave=False)
        
        
        for step, batch in enumerate(epoch_iterator):
            batch = {k:v.to(device) for k,v in batch.items()}
            # forward pass
            rank_output, rating_pred, enc_loss, dec_rmse = model(batch)

            rating_mask = (batch['item_rating'] != 0)            
            org_loss = metrics.RMSE(rating_pred, batch['item_rating'], rating_mask)
            org_losses.update(org_loss)
            y_rank_value = F.softmax(batch['anchor_ratings'].float(), dim=-1)
            # rank_logits = F.softmax(rank_output, dim=-1)
            # rank_loss = RMSE(rank_logits, y_rank_value) # 추후에, 하나로 합친 결과에 대한 loss계산하는 방식으로 추가 실험
            rank_loss = metrics.BPR(rank_output, batch['anchor_ratings'].float()) # 추후에, 하나로 합친 결과에 대한 loss계산하는 방식으로 추가 실험
            rank_losses.update(rank_loss)
            
            loss = org_loss + rank_loss
            # loss = org_loss
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss)
            epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), losses.val))
            
        # validation
        if device.type=='cuda':
            end.record(stream)
            torch.cuda.synchronize()
            
        total_time += (start.elapsed_time(end))
        valid_loss, best_rmse, best_mae, best_ndcg, valid_ndcg, valid_rmse, valid_mae, update_cnt = valid(model, ds_iter, epoch, checkpoint_path, step, best_rmse, best_mae, best_ndcg, update_cnt)
        lr_scheduler.step(valid_loss) # ReduceLROnPlateau

        # Tensorboard recording
        writer.add_scalars('Loss', {'Train':losses.avg, 'Valid':valid_loss,}, epoch)
        writer.add_scalar('RMSE/Test', valid_rmse, epoch)
        writer.add_scalar('MAE/Test', valid_mae, epoch)

        print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Rank Loss: {rank_losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch NDCG@10: {valid_ndcg:.4f} || epoch RMSE: {valid_rmse:.4f} || epoch MAE: {valid_mae:.4f} || best RMSE: {best_rmse:.4f} || best MAE: {best_mae:.4f} || best NDCG@10: {best_ndcg:.4f} ||\n")
        if epoch > 100:
            break
        if update_cnt > 20: 
            break
    writer.close()

    print('\n [Train Finished]')
    print("total training time (s): {}".format((time.time()-init_t)))
    print("total training time (ms): {}".format(total_time))
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("total memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    print(torch.cuda.memory_summary(device=device.index))
    
def eval2(model, ds_iter):
    model.eval()
    metrics = Metrics()
    if device.type=='cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device=device)
        start.record(stream)
        
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
            
            rank_output, rating_pred, _, _ = model(batch, is_train=False)
            mask = (batch['item_rating'] != 0)
            rmse = metrics.RMSE(rating_pred, batch['item_rating'], mask).item()
            mae = metrics.MAE(rating_pred, batch['item_rating'], mask)
            total_rmse += rmse
            total_mae += mae
            
            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), rmse))
            
            # zero padding인 경우 제외
            mask = (batch['anchor_items']!=0)
            rank_output = rank_output*mask
            
            item_lst, rating_lst, output_lst = [],[],[]
            for i in range(batch['anchor_user'].size(0)):
                items = batch['anchor_items'][i][batch['anchor_items'][i]!=0].data.cpu().tolist()
                ratings = batch['anchor_ratings'][i][batch['anchor_ratings'][i]!=0].data.cpu().tolist()
                outputs = rank_output[i][rank_output[i]!=0].data.cpu().tolist()
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
    output_df['logits'] = output_df['outputs'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())
    output_df['targets'] = output_df['anchor_ratings'].map(lambda x:F.softmax(torch.tensor(x, dtype=torch.float), dim=-1).tolist())    
    output_df[['ndcg@10','precision@10','recall@10', 'rec@10', 'ideal@10']] = output_df.apply(lambda x:metrics.rank_metrics(x['anchor_items'], x['logits'], x['anchor_ratings'], 10), axis=1)
    output_df = output_df.dropna(how='any')
    filtered_ndcg = output_df[output_df['anchor_items'].apply(len)>=10]['ndcg@10'].mean()
    total_ndcg = output_df['ndcg@10'].mean()
    total_precision = output_df['precision@10'].mean()
    total_recall = output_df['recall@10'].mean()
    total_rmse /= (step+1)
    total_mae /= (step+1)           
    
    output_df.to_csv(f'eval_output_{args.dataset}_{args.item_per_user}.csv', index=False)
        
    if device.type=='cuda':
        end.record(stream)
        torch.cuda.synchronize()

    print("\n [Evaluation Results]")
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print("TOTAL RECALL@10: %2.5f" % total_recall)
    print("TOTAL PRECISION@10: %2.5f" % total_precision)
    print("TOTAL NDCG@10: %2.5f" % total_ndcg)
    print("Filtered NDCG@10: %2.5f" % filtered_ndcg)
    # print(f"Precision@5 : {total_precision_5} / Recall@5 : {total_recall_5} / NDCG@5 : {total_ndcg_5}")
    # print(f"Precision@10 : {total_precision_10} / Recall@10 : {total_recall_10} / NDCG@10 : {total_ndcg_10}")
    print(f"total eval time: {(start.elapsed_time(end))}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))
    
def get_args():
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--device", type=str, default='single')
    parser.add_argument("--id", type=int, default=0)
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
    parser.add_argument("--test_ratio", type=float, default=0.1, help="percentage of valid/test dataset")
    parser.add_argument('--user_seq_len', type=int, default=30, help="user random walk sequence length")
    parser.add_argument('--item_per_user', type=int, default=5, help="number of items per user")
    parser.add_argument('--return_params', type=int, default=1, help="return param value for generating random sequence")
    parser.add_argument('--augs', type=int, default=1, help="how many times augment train data per anchor user")
    parser.add_argument('--regen', type=str, default='no', help="Whether regen dataframe(random walk & total df) or not")    
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

    ### get model config ###
    model_config = Config[args.dataset]["model"]
    training_config = Config[args.dataset]["training"]
    # batch size update
    training_config["batch_size"] = args.bs
    
    # dataset & dataloader
    train_ds = MyDataset(total_train)
    valid_ds = MyDataset(total_valid)
    test_ds = MyDataset(total_test)
    
    ds_iter = {
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=1), 
            "valid":DataLoader(valid_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=1),
            "test":DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=1)
    }

    ######################################################### model initialization #########################################################

    training_config["learning_rate"] = args.lr
    # model config - num users & num items & degrees
    model_config["num_user"] = data_making.num_user
    model_config["num_item"] = data_making.num_item
    model_config["max_user_degree"] = data_making.max_user_degree
    model_config["max_item_degree"] = data_making.max_item_degree
    # model expansion (1) : Increase # of Encoder/Decoder Blocks
    model_config["num_layers_enc"] = args.num_layers_enc + int(math.log(args.augs,2))
    model_config["num_layers_dec"] = args.num_layers_dec + int(math.log(args.augs,2))
    
    # model expansion (2) : MoE topk router
    model_config["n_experts"] = args.n_experts
    # model expansion (2)-2 : MoE topk # of experts
    model_config["topk"] = args.topk + int(math.log(args.augs,2))
    
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
    name_train_augs = str(args.augs)
    name_test_augs = str(str(min(3,args.augs)) if args.augs else '')
    args.name = '_'.join([name_dataset, name_seed, name_u_len, name_i_len, name_n_enc, name_n_dec, name_train_augs, name_test_augs])
    checkpoint_path = os.path.join(checkpoint_dir, f'{args.name}.model') # set model name
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"])

    training_config["num_train_steps"] = len(ds_iter['train'])

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        mode = 'min',
        factor = 0.85,
        patience = 3,
        threshold = 1e-2,
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