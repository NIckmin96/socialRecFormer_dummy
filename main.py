import os
import sys
import logging
import argparse
import random
import math
import time
import pynvml
from tqdm import tqdm
import numpy as np
import pandas as pd
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
from dataset import RWDataset, AnchorDataset
from models.transformer import Transformer
from scheduler import WarmupCosineSchedule

# Ray Tune
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune.search.optuna import OptunaSearch

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

def valid(model, ds_iter, epoch, checkpoint_path, global_step, best_loss, best_ndcg, update_cnt):
    eval_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        # FIXME: valid를 기준으로 저장 X, test를 기준으로 바로 저장. 
        epoch_iterator = tqdm(zip(*ds_iter['valid']),
                              desc="Validating",
                              ascii=" =",
                              leave=True)
        ndcg_df = pd.DataFrame()
        total_step = len(ds_iter['valid'][0])
        for step, (batch1, batch2) in enumerate(epoch_iterator):
            # 첫번째 batch : bs x n x d 
            batch1['rw_seq'] = batch1['rw_seq'].to(device)
            batch1['rw_degree'] = batch1['rw_degree'].to(device)
            # 첫번째 batch : bs x i x d (zero-padding 포함)
            batch2['user'] = batch2['user'].to(device)
            batch2['user_degree'] = batch2['user_degree'].to(device)
            batch2['product'] = batch2['product'].to(device)
            batch2['product_degree'] = batch2['product_degree'].to(device)
            batch2['ratings'] = batch2['ratings'].to(device)

            loss, target, logits = model(batch1, batch2)
            # rmse loss 계산(Rating)
            
            eval_losses.update(loss)
            
            # rank metric                
            df = pd.DataFrame({'users':batch2['user'].data.cpu().tolist(),
                                'items':batch2['product'].data.cpu().tolist(),
                                'ratings':target.data.cpu().tolist(),
                                'logits':logits.data.cpu().tolist()})
            ndcg_df = pd.concat([ndcg_df, df], axis=0)                
            
            epoch_iterator.set_description(
                        "Validating (%d / %d Steps) (loss=%2.5f)" % (step, total_step, loss.item()))
        
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
        
    if ((1/loss.item())*0.4+(total_ndcg)*0.6 > (1/best_loss)*0.4+(best_ndcg)*0.6): 
        best_ndcg = total_ndcg
        best_loss = loss.item()
        torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
        print(f'\t best model saved: step = {global_step}, epoch = {epoch}, Best RMSE = {loss.item():.6f}, Best NDCG@10 = {total_ndcg.item():.6f}')
        update_cnt = 0
        
    else:
        update_cnt += 1

    return eval_losses.avg, best_loss, best_ndcg, update_cnt

def train(model, optimizer, lr_scheduler, ds_iter, training_config, writer):

    logger.info("***** Running training *****")

    checkpoint_path = training_config['checkpoint_path']
    best_loss = 1e10
    best_ndcg = 0

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
        epoch_iterator = tqdm(zip(*ds_iter['train']),
                            desc="Training",
                            ascii=" =",
                            leave=True)
        
        total_step = len(ds_iter['train'][0])
        print("total step:", total_step)
        for step, (batch1, batch2) in enumerate(epoch_iterator):
            # 첫번째 batch : bs x n x d 
            batch1['rw_seq'] = batch1['rw_seq'].to(device)
            batch1['rw_degree'] = batch1['rw_degree'].to(device)
            # 첫번째 batch : bs x i x d (zero-padding 포함)
            batch2['user'] = batch2['user'].to(device)
            batch2['user_degree'] = batch2['user_degree'].to(device)
            batch2['product'] = batch2['product'].to(device)
            batch2['product_degree'] = batch2['product_degree'].to(device)
            batch2['ratings'] = batch2['ratings'].to(device)

            # forward pass
            loss, target, logits = model(batch1, batch2)
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss)
            epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, total_step, losses.val))
            
        # validation
        if device.type=='cuda':
            end.record(stream)
            torch.cuda.synchronize()
            
        total_time += (start.elapsed_time(end))
        valid_loss, best_loss, best_ndcg, update_cnt = valid(model, ds_iter, epoch, checkpoint_path, step, best_loss, best_ndcg, update_cnt)
        lr_scheduler.step(valid_loss) # ReduceLROnPlateau

        print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Test Loss: {valid_loss:.4f} || best loss: {best_loss:.4f} || best NDCG@10: {best_ndcg:.4f}\n")
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
    
def eval2(model, ds_iter):
    model.eval()
    if device.type=='cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device=device)
        start.record(stream)
        
        epoch_iterator = tqdm(zip(*ds_iter['test']),
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
        total_step = len(ds_iter['test'][0])
        with torch.no_grad():
            for step, (batch1, batch2) in enumerate(epoch_iterator):     
                # 첫번째 batch : bs x n x d 
                batch1['rw_seq'] = batch1['rw_seq'].to(device)
                batch1['rw_degree'] = batch1['rw_degree'].to(device)
                # 두번째 batch : bs x i x d (zero-padding 포함)
                batch2['user'] = batch2['user'].to(device)
                batch2['user_degree'] = batch2['user_degree'].to(device)
                batch2['product'] = batch2['product'].to(device)
                batch2['product_degree'] = batch2['product_degree'].to(device)
                batch2['ratings'] = batch2['ratings'].to(device)
                
                loss, target, logits = model(batch1, batch2)
                
                # rank metric                
                df = pd.DataFrame({'users':batch2['user'].data.cpu().tolist(),
                                   'items':batch2['product'].data.cpu().tolist(),
                                   'ratings':target.data.cpu().tolist(),
                                   'logits':logits.data.cpu().tolist()})
                ndcg_df = pd.concat([ndcg_df, df], axis=0)                
            
                epoch_iterator.set_description(
                            "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, total_step, loss.item()))
                
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
        # ndcg_df.to_csv(f'./ndcg_test_{args.dataset}.csv', index=False)
                
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
    # model config
    parser.add_argument('--num_layers_enc', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_ffn', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=int, default=0.3)
    parser.add_argument('--n_experts', type=int, default=8)
    parser.add_argument('--topk', type=int, default=2)
    # train config
    parser.add_argument('--lr', type=float, default=1e-3) # rating 기준 rw 생성의 경우 default = 1e-3
    parser.add_argument('--weight_decay', type=float, default=0.03) # rating 기준 rw 생성의 경우 default = 1e-3
    # dataset args
    parser.add_argument("--dataset", type = str, default="epinions", help = "ciao, epinions")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="percentage of valid/test dataset")
    parser.add_argument('--seq_len', type=int, default=200, help="user random walk sequence length")
    parser.add_argument('--regen', type=bool, default=False, help="Whether regenerate dataframe(random walk & total df) or not")    
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
    if args.regen:
        print("Re-Creating Datatset...")
    else:
        print("Loading Datatset...")
    data_making = dm.DatasetMaking(args)
    
    print("\n")
    total_train = data_making.total_train
    total_valid = data_making.total_valid
    total_test = data_making.total_test

    train_ds, train_ds2 = RWDataset(total_train), AnchorDataset(total_train)
    valid_ds, valid_ds2 = RWDataset(total_valid), AnchorDataset(total_valid)
    test_ds, test_ds2 = RWDataset(total_test), AnchorDataset(total_test)


    ### get model config ###
    model_config = {}
    training_config = Config[args.dataset]["training"]
    # batch size update
    training_config["batch_size"] = args.bs
    
    def zero_padding(batch):
        padded_batch = {'user':torch.stack([data['user'].unsqueeze(0) for data in batch]),
                        'user_degree':torch.stack([data['user_degree'].unsqueeze(0) for data in batch]),
                        'product':pad_sequence([data['product'] for data in batch], batch_first=True, padding_value=0),
                        'product_degree':pad_sequence([data['product_degree'] for data in batch], batch_first=True, padding_value=0),
                        'ratings':pad_sequence([data['ratings'].squeeze(0) for data in batch], batch_first=True, padding_value=0)}
        # print(padded_batch['user'].shape, padded_batch['product'].shape, padded_batch['product_degree'].shape, padded_batch['ratings'].shape)
        return padded_batch
    
    ds_iter = {
            "train":[DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=0),
                     DataLoader(train_ds2, batch_size = training_config["batch_size"], shuffle=False, num_workers=0, collate_fn=zero_padding)],
            
            "valid":[DataLoader(valid_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=0),
                     DataLoader(valid_ds2, batch_size = training_config["batch_size"], shuffle=False, num_workers=0, collate_fn=zero_padding)],
            
            "test":[DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=0),
                    DataLoader(test_ds2, batch_size = training_config["batch_size"], shuffle=False, num_workers=0, collate_fn=zero_padding)]
    }

    ######################################################### model initialization #########################################################

    training_config["learning_rate"] = args.lr
    # model config - num users & num items & degrees
    model_config["seq_len"] = args.seq_len
    model_config["max_node"] = data_making.num_user+data_making.num_item
    model_config["max_degree"] = max(data_making.max_user_degree, data_making.max_item_degree)
    model_config['d_model'] = args.d_model
    model_config['d_ffn'] = args.d_ffn
    model_config['num_heads'] = args.num_heads
    model_config['dropout'] = args.dropout
    model_config["num_layers_enc"] = args.num_layers_enc
    model_config["n_experts"] = args.n_experts
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
    
    name_dataset = str(args.dataset)
    name_seed = str(args.seed)
    name_len = str(args.seq_len)
    name_n_enc = str(model_config['num_layers_enc'])
    args.name = '_'.join([name_dataset, name_seed, name_len, name_n_enc])
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
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"GPU index: {device.index}")
    print("\n")
    
    model = model.to(device)

    ############################################################ training preparation ############################################################

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = args.lr,
        betas=(0.9, 0.999), eps=1e-6, weight_decay=args.weight_decay
    )

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