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
from utils import redirect_stdout
from config import Config
from dataset import MyDataset
from models.transformer import Transformer
from scheduler import WarmupCosineSchedule
import requests

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

# def MaskedMSELoss(target, prediction):
#     """
#     Compute Masked MSELoss
#     """
#     mask = (target != 0).float()
#     squared_diff = (prediction - target)**2 * mask
#     loss = torch.sum(squared_diff) / torch.sum(mask)

#     return loss

def valid(model, ds_iter, epoch, checkpoint_path, global_step, best_dev_rmse, best_dev_mae, init_t, update_cnt):
    # val_rmse = []
    # val_mae = []
    criterion = nn.MSELoss()
    eval_losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        # FIXME: valid를 기준으로 저장 X, test를 기준으로 바로 저장. 
        epoch_iterator = tqdm(ds_iter['test'],
                              desc="Validating (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              leave=False)
        pred, trg, msk = [], [], []
        for step, batch in enumerate(epoch_iterator):
            batch['user_seq'] = batch['user_seq'].to(device)
            batch['user_degree'] = batch['user_degree'].to(device)
            batch['item_list'] = batch['item_list'].to(device)
            batch['item_degree'] = batch['item_degree'].to(device)
            batch['item_rating'] = batch['item_rating'].to(device)
            batch['spd_matrix'] = batch['spd_matrix'].to(device)

            outputs, enc_loss, dec_loss = model(batch, is_train=False)

            # loss = criterion(outputs.float(), batch['item_rating'].float())
            # FIXME:
                # 현재 target(batch['item_rating])은 0이 많이 포함되어 있는 sparse한 rating matrix로, shape가 [seq_len_user, seq_len_item] (batch dim 제외)
                # model output 또한 마찬가지로 shape가 [seq_len_user, seq_len_item].
                # 단순히 이 둘의 MSELoss를 계산하는 경우, known rating에 대한 제곱오차만을 계산하는 것이 아닌 unknown rating(0)에 대한 제곱오차도 계산하게 됨.
                # 따라서 Masked MSELoss를 사용.
                # model의 출력에서 unknown rating에 대한 부분을 0으로 masking 처리, 제곱오차 계산 시 known rating과 만의 제곱오차를 계산.
            
            #print(batch['item_rating'].shape)
            batch['item_rating'] = batch['item_rating'][:,0] 
            #print(batch['item_rating'].shape)
            outputs = outputs[:,0]
            #outputs = torch.mean(outputs,dim=1)
            #print(outputs.shape)
            
            mask = (batch['item_rating'] != 0)
            squared_diff = (outputs - batch['item_rating'])**2 * mask
            loss = torch.sum(squared_diff) / torch.sum(mask)
            #loss = criterion(outputs[mask].float(),batch['item_rating'][mask].float()).cuda()

            loss += enc_loss
            loss += dec_loss
            
            eval_losses.update(loss)
            
            # 실제 rating matrix에서 0이 아닌 부분(실제 매긴 rating)과만 loss를 계산
                # -> 현재 목표는 rating regression이기 때문이니까.
            # mse = F.mse_loss(outputs[mask].float(), batch['item_rating'][mask].float(), reduction='none')
            # rmse = torch.sqrt(mse.mean())
            # mae = F.l1_loss(outputs[mask].float(), batch['item_rating'][mask].float(), reduction='mean'

            # val_rmse.append(rmse)
            # val_mae.append(mae)
            
            pred.append(outputs)
            trg.append(batch['item_rating'])
            msk.append(mask)

            epoch_iterator.set_description(
                        "Validating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), eval_losses.val))

        # if total_rmse < best_dev_rmse:
        #     best_dev_rmse = total_rmse
        #     torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
        #     print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}')
            
        # if total_mae < best_dev_mae:
        #     best_dev_mae = total_mae
        #     torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
        #     print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}')
        
        pred = torch.cat(pred)
        trg = torch.cat(trg)
        msk = torch.cat(msk)
        mse = F.mse_loss(pred[msk].float(), trg[msk].float(), reduction='none')
        total_rmse = torch.sqrt(mse.mean())
        total_mae = F.l1_loss(pred[msk].float(), trg[msk].float(), reduction='mean')
        rmse = torch.sqrt(torch.mean(torch.pow((pred[msk].float() - trg[msk].float()), 2)))

        # baseline의 metric보다 낮은 경우
        if (torch.mean(total_rmse, dim=0).item()<baseline_rmse) & (torch.mean(total_mae, dim=0).item()<baseline_mae):
            # best보다 낮은 경우
            if (total_rmse+total_mae < best_dev_rmse+best_dev_mae): 
                best_dev_rmse = total_rmse
                best_dev_mae = total_mae
                torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
                print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}')
                update_cnt = 0
            # best보다 높지만, best가 baseline보다 높은 경우 update
            elif (torch.mean(best_dev_rmse, dim=0).item()>baseline_rmse) | (torch.mean(best_dev_mae, dim=0).item()>baseline_mae): 
                best_dev_rmse = total_rmse
                best_dev_mae = total_mae
                torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
                print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}')
                update_cnt = 0
            else:
                update_cnt += 1

        # baseline보다 높지만, best보다 낮은 경우
        elif total_rmse+total_mae < best_dev_rmse+best_dev_mae:
            best_dev_rmse = total_rmse
            best_dev_mae = total_mae
            torch.save({"model_state_dict":model.state_dict()}, checkpoint_path)
            print(f'\t best model saved: step = {global_step}, epoch = {epoch}, test RMSE = {total_rmse.item():.6f}, test MAE = {total_mae.item():.6f}')
            update_cnt = 0
        else:
            update_cnt += 1

    # return eval_losses.avg, best_dev_rmse, best_dev_mae, total_rmse, total_mae, update_cnt
    return eval_losses.val, best_dev_rmse, best_dev_mae, total_rmse, total_mae, update_cnt

def train(model, optimizer, lr_scheduler, ds_iter, training_config, writer):
    global baseline_rmse, baseline_mae

    # TODO: Epoch당 loss, RMSE, MAE 추적 => TensorBoard 또는 파일 저장을 통해 tracing할 수 있도록.
    logger.info("***** Running training *****")
    logger.info("  Total steps = %d", training_config["num_train_steps"])

    checkpoint_path = training_config['checkpoint_path']
    best_dev_rmse = 9999.0
    best_dev_mae = 9999.0
    baseline_rmse = training_config['baseline_rmse']
    baseline_mae = training_config['baseline_mae']

    total_epochs = training_config["num_epochs"]

    model.train()
    init_t = time.time()
    total_time = 0
    update_cnt = 0
    criterion = nn.MSELoss()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device=device)
    start.record(stream)

    # Training step
    for epoch in range(total_epochs):
        lr_lst = []
        losses = AverageMeter()
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
            batch['spd_matrix'] = batch['spd_matrix'].to(device)

            # forward pass
            outputs, enc_loss, dec_loss = model(batch)

            # compute loss
                        
            ####### dec_loss
            mask = (batch['item_rating'] != 0)
            squared_diff = (outputs - batch['item_rating'])**2 * mask
            org_loss = torch.sum(squared_diff) / torch.sum(mask)
            org_loss = torch.sqrt(org_loss) # RMSE

            batch['item_rating'] = batch['item_rating'][:,0] 
            outputs = outputs[:,0]
            
            mask = (batch['item_rating'] != 0)
            squared_diff = (outputs - batch['item_rating'])**2 * mask
            new_loss = torch.sum(squared_diff) / torch.sum(mask)

            # loss =  org_loss + new_loss * training_config["alpha"] + dec_loss * training_config["gamma"] + enc_loss * training_config["beta"] 
            
            # [ORG]
            loss = org_loss*training_config["alpha"] + dec_loss * training_config["gamma"] + enc_loss * training_config["beta"]
            
            # [DEV]
            # loss = org_loss*training_config["alpha"] + enc_loss * training_config["beta"] 
            loss.backward()

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping
            optimizer.step()
            optimizer.zero_grad()

            losses.update(loss)
            epoch_iterator.set_description(
                        "Training (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), losses.val))
            
        # validation
        end.record(stream)
        torch.cuda.synchronize()
        total_time += (start.elapsed_time(end))
        valid_loss, best_dev_rmse, best_dev_mae, valid_rmse, valid_mae, update_cnt = valid(model, ds_iter, epoch, checkpoint_path, step, best_dev_rmse, best_dev_mae, init_t, update_cnt)
        lr_scheduler.step(valid_loss) # ReduceLROnPlateau
        # lr_scheduler.step() # cosineannealinglr
        model.train()
        start.record(stream)

        # Tensorboard recording
        writer.add_scalars('Loss', {'Train':losses.avg, 'Valid':valid_loss,}, epoch)
        writer.add_scalar('RMSE/Test', valid_rmse, epoch)
        writer.add_scalar('MAE/Test', valid_mae, epoch)

        print(f"Epoch {epoch:03d}: Train Loss: {losses.avg:.4f} || Test Loss: {valid_loss:.4f} || epoch RMSE: {valid_rmse:.4f} || epoch MAE: {valid_mae:.4f} || best RMSE: {best_dev_rmse:.4f} || best MAE: {best_dev_mae:.4f}")
        if epoch > 100:
            break
        # if update_cnt > 10:
        if update_cnt > 20: # dev
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
        # for _, batch in ds_iter['test']:
        pred, trg, msk = [], [], []
        for step, batch in enumerate(epoch_iterator):
            
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].to(device)
            batch['user_degree'] = batch['user_degree'].to(device)
            batch['item_list'] = batch['item_list'].to(device)
            batch['item_degree'] = batch['item_degree'].to(device)
            batch['item_rating'] = batch['item_rating'].to(device)
            batch['spd_matrix'] = batch['spd_matrix'].to(device)
            outputs, enc_loss, dec_loss = model(batch, is_train=False)

            # FIXME: 
                # 현재 target(batch['item_rating'])은 0이 많이 포함되어 있는 sparse한 rating matrix로, shape가 [seq_len_user, seq_len_item] (batch dim 제외)
                # model output 또한 마찬가지로 shape가 [seq_len_user, seq_len_item].
                # 단순히 이 둘의 MSELoss를 계산하는 경우, known rating에 대한 제곱오차만을 계산하는 것이 아닌 unknown rating(0)에 대한 제곱오차도 계산하게 됨.
                # 따라서 Masked MSELoss를 사용
                # model의 출력에서 unknown rating에 대한 부분을 0으로 masking 처리, 제곱오차 계산 시 known rating과 만의 제곱오차를 계산하게 된다.
            batch['item_rating'] = batch['item_rating'][:,0] 
            outputs = outputs[:,0]            
            mask = (batch['item_rating'] != 0)
            squared_diff = (outputs - batch['item_rating'])**2 * mask
            loss = torch.sum(squared_diff) / torch.sum(mask)
            loss += enc_loss
            # loss += dec_loss
            eval_losses.update(loss)
            
            pred.append(outputs)
            trg.append(batch['item_rating'])
            msk.append(mask)

            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), eval_losses.val))
        
        pred = torch.cat(pred)
        trg = torch.cat(trg)
        msk = torch.cat(msk)
        print(pred[msk])
        print(trg[msk])
        mse = F.mse_loss(pred[msk].float(), trg[msk].float(), reduction='none')
        total_rmse = torch.sqrt(mse.mean())
        total_mae = F.l1_loss(pred[msk].float(), trg[msk].float(), reduction='mean')
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')

    end.record(stream)
    torch.cuda.synchronize()

    print("\n [Evaluation Results]")
    print("Loss: %2.5f" % eval_losses.avg)
    print("RMSE: %2.5f" % total_rmse)
    print("MAE: %2.5f" % total_mae)
    print(f"total eval time: {(start.elapsed_time(end))}")
    print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))
    print("all memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.allocated']>>20))

    
def get_args():
    parser = argparse.ArgumentParser(description='Transformer for Social Recommendation')
    parser.add_argument("--mode", type = str, default="train",
                        help="train eval")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, help="checkpoint model name")
    parser.add_argument('--num_layers_enc', type=int, default=4, help="num enc layers")
    parser.add_argument('--num_layers_dec', type=int, default=4, help="num dec layers")
    parser.add_argument('--n_experts', type=int, default=8, help="MoE number of total experts")
    parser.add_argument('--topk', type=int, default=1, help="MoE number of routers")
    parser.add_argument('--lr', type=float, default=1e-4)
    # dataset args
    parser.add_argument("--dataset", type = str, default="epinions", help = "ciao, epinions")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="percentage of valid/test dataset")
    parser.add_argument('--user_seq_len', type=int, default=30, help="user random walk sequence length")
    parser.add_argument('--item_per_user', type=int, default=5, help="number of items per user")
    parser.add_argument('--return_params', type=int, default=1, help="return param value for generating random sequence")
    parser.add_argument('--train_augs', type=int, default=10, help="how many times augment train data per anchor user")    
    parser.add_argument('--test_augs', type=bool, default=False, help="Whether augment test data set in proportion to train_augs or not / max = 3")    
    parser.add_argument('--regenerate', type=bool, default=False, help="Whether regenerate dataframe(random walk & total df) or not")    
    parser.add_argument('--bs', type=int, default=128, help="Batch size of dataloader")
    
    args = parser.parse_args()
    return args

def main():
    global device

    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    ### get model config ###
    model_config = Config[args.dataset]["model"]
    training_config = Config[args.dataset]["training"]

    training_config["learning_rate"] = args.lr
    # model expansion (1) : Increase # of Encoder/Decoder Blocks
    model_config["num_layers_enc"] = int(math.log(args.train_augs+1)*args.num_layers_enc)
    model_config["num_layers_dec"] = int(math.log(args.train_augs+1)*args.num_layers_dec)
    
    # model expansion (2) : MoE topk router
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
    print(model)

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
            device = 'cpu'

    pynvml.nvmlShutdown()

    if device=='cpu': 
        raise DeviceError
    # tmp
    elif torch.cuda.device_count()>1:
        device = torch.device('cuda:2')
    else:
        device = torch.device('cuda:0')
    
    print(f"GPU index: {device.index}")
    print("\n")
    
    model = model.to(device)

    ######################################################### data preparation #########################################################

    ### FIXME: 전체 데이터에 대해 파일 생성이 오래 걸림 (현재 시퀀스의 rating matrix 생성하는 부분이 문제로 보임)
        ### FIXME: (231012) validation set을 통해 모델이 잘 train 되는것은 확인했으므로, 바로 test를 진행하면서 model을 저장.

    '''
    data_making_2.py에서는 한번에 train/valid/test를 만들고 있기 때문에, MyDataset 내에서 DatasetMaking 객체를 따로 불러오면 비효율적임
    
    1. 파일 있는지 확인(train/valid/test 전부)
    2. 없으면 DatasetMaking 객체 생성
    3. DatasetMaking 객체 내 train/valid/test -> dataset

    regenerate 부분 추가하기
    - regenerate==True시, 전체 데이터 처음부터 다시 생성하기

    train data 생성시에, valid/test 데이터는 그대로 유지시키기
    '''
    # regenerate 여부 확인
    if args.regenerate:
        print("Re-Creating Datatset...")
    else:
        print("Loading Datatset...")
    data_making = dm.DatasetMaking(args.dataset, args.seed, args.user_seq_len, args.item_per_user, args.return_params, args.train_augs, args.test_augs, args.regenerate)    

    # 1. file path check(train_augs & test_augs)
    train_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                              f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_train_{args.train_augs}times.pkl')
    # valid_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
    #                           f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{args.item_seq_len}_rp_{args.return_params}_valid.pkl')
    if args.test_augs:
        print(f"dataset : {args.dataset}\n seed : {args.seed}\n test_ratio: {args.test_ratio}\n user_seq_len : {args.user_seq_len}\n item_seq_len : {name_i_len}\n return_params : {args.return_params}\n train_augs : {args.train_augs}\n test_augs : {args.train_augs}\n \
            num_enc_layers : {name_n_enc}\n num_dec_layers : {name_n_dec}")
        test_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                                f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_test_{args.train_augs}times.pkl')
    else:
        print(f"dataset : {args.dataset}\n seed : {args.seed}\n test_ratio: {args.test_ratio}\n user_seq_len : {args.user_seq_len}\n item_seq_len : {name_i_len}\n return_params : {args.return_params}\n train_augs : {args.train_augs}\n \
            num_enc_layers : {name_n_enc}\n num_dec_layers : {name_n_dec}")
        test_path = os.path.join(os.getcwd(), 'dataset', args.dataset, 
                                f'sequence_data_seed_{args.seed}_walk_{args.user_seq_len}_itemlen_{name_i_len}_rp_{args.return_params}_test.pkl')
    
    # print(f"dataset : {args.dataset}\n seed : {args.seed}\n test_ratio: {args.test_ratio}\n user_seq_len : {args.user_seq_len}\n item_seq_len : {args.item_seq_len}\n return_params : {args.return_params}\n train_augs : {args.train_augs}")
    print("\n")
    # data_making = dm.DatasetMaking(args.dataset, args.seed, args.user_seq_len, args.item_seq_len, args.return_params, args.train_augs, args.test_augs, args.regenerate)
    total_train = data_making.total_train
    # total_valid = data_making.total_valid
    total_test = data_making.total_test        

    train_ds = MyDataset(total_train)
    # valid_ds = MyDataset(total_valid)
    test_ds = MyDataset(total_test)

    # batch size update
    training_config["batch_size"] = args.bs
    ds_iter = {
            # "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=8),
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            # "dev":DataLoader(dev_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            # "test":DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=8)
            "test":DataLoader(test_ds, batch_size = training_config["batch_size"], shuffle=False, num_workers=4)
    }

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
    )

    # total_steps는 cycle당 있는 step 수. 없다면 epoch와 steps_per_epoch를 전댈해야함.
        # steps_per_epoch는 한 epoch에서의 전체 step 수: (total_number_of_train_samples / batch_size)
    total_epochs = training_config["num_epochs"]
    total_train_samples = len(train_ds)
    # training_config["num_train_steps"] = math.ceil(total_train_samples / total_epochs) # why divisor = 'total_epochs' not 'batch_size'???
    training_config["num_train_steps"] = len(ds_iter['train'])
    

    # lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-4, mode='triangular', step_size_up=5, cycle_momentum=False, verbose=True)
    
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7, verbose=True)


    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer = optimizer,
        mode = 'min',
        factor = 0.8,
        patience = 5,
        threshold = 3e-2,
        min_lr = 1e-6,
        verbose = True
    )
    
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR( # [CHECK]
    #     optimizer = optimizer,
    #     max_lr = training_config["learning_rate"],
    #     # pct_start = training_config["warmup"] / training_config["num_train_steps"], # 40/batch개수
    #     pct_start = 0.15,
    #     # anneal_strategy = training_config["lr_decay"],
    #     anneal_strategy = 'cos',
    #     epochs = training_config["num_epochs"],
    #     # steps_per_epoch = 2 * len(ds_iter['train'])
    #     steps_per_epoch = len(ds_iter['train'])
    # )


    ### TensorBoard writer preparation ###
    writer = SummaryWriter(os.path.join(log_dir,f"{args.name}.tensorboard"))
    ### train ###
    if args.mode == 'train':
        train(model, optimizer, lr_scheduler, ds_iter, training_config, writer)

    # Since train logging is done by TensorBoard, log only test result.
    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.name))
    redirect_stdout(open(log_path, 'w'))

    print(json.dumps(args.__dict__, indent = 4))

    print(json.dumps([model_config, training_config], indent = 4))

    # print(model)

    ### eval ###
    print(checkpoint_path)
    if os.path.exists(checkpoint_path): #and checkpoint_path != os.getcwd() + '/checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
        eval(model, ds_iter)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()