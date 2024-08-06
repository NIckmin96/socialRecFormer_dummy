import os
import sys
import logging
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loss import BPRLoss
from utils import redirect_stdout
from config import Config
from tmp.dataset import MyDataset
from models.transformer import Transformer
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import pandas as pd
from ast import literal_eval    # convert str type list to original type
logger = logging.getLogger(__name__)

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

def eval(model, ds_iter):

    val_rmse = []
    val_mae = []
    eval_losses = AverageMeter()
    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        epoch_iterator = tqdm(ds_iter['test'],
                        desc="Validating (X / X Steps) (loss=X.X)",
                        bar_format="{l_bar}{r_bar}",
                        dynamic_ncols=True,
                        leave=False)
        # for _, batch in ds_iter['test']:
        for step, batch in enumerate(epoch_iterator):
            #print(batch.keys())
            # 모델의 입력은 batch 그 자체, batch는 Dict이며 따라서 Dict 안의 tensor들을 device로 load.
            batch['user_seq'] = batch['user_seq'].cuda()
            batch['user_degree'] = batch['user_degree'].cuda()
            batch['item_list'] = batch['item_list_full'].cuda()
            batch['item_degree'] = batch['item_degree_full'].cuda()
            batch['item_rating'] = batch['item_rating_full'].cuda()
            batch['spd_matrix'] = batch['spd_matrix'].cuda()

            outputs, user_emb, item_emb = model(batch, True)
            outputs = outputs.squeeze()

            mask = (batch['item_rating'][0] != 0)
            squared_diff = (outputs - batch['item_rating'][0])**2 * mask
            loss = torch.sum(squared_diff) / torch.sum(mask)

            # eval_losses.update(loss.mean())
            eval_losses.update(loss)
            
            # 실제 rating matrix에서 0이 아닌 부분(실제 매긴 rating)과만 loss를 계산
                # -> 현재 목표는 rating regression이기 때문이니까.

            mse = F.mse_loss(outputs, batch['item_rating'][0], reduction='none')
            rmse = torch.sqrt(mse.mean())
            mae = F.l1_loss(outputs, batch['item_rating'][0], reduction='mean')

            val_rmse.append(rmse)
            val_mae.append(mae)

            epoch_iterator.set_description(
                        "Evaluating (%d / %d Steps) (loss=%2.5f)" % (step, len(epoch_iterator), eval_losses.val))

        total_rmse = sum(val_rmse) / len(val_rmse)
        total_mae = sum(val_mae) / len(val_mae)

    end.record()
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
    parser.add_argument("--dataset", type = str, default="epinions",
                        help = "ciao, epinions")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--name', type=str, help="checkpoint model name")
    parser.add_argument('--user_seq_len', type=int, default=20, help="user random walk sequence length")
    parser.add_argument('--item_seq_len', type=int, default=50, help="item list length")
    parser.add_argument('--topk', type=int, default=5, help="evaluation top k")
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    ### get model config ###
    model_config = Config[args.dataset]["model"]
    training_config = Config[args.dataset]["training"]
    #training_config["learning_rate"] = args.learning_rate

    ### log preparation ###
    log_dir = os.getcwd() + f'/logs/log_seed_{args.seed}/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.dataset)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.name))
    redirect_stdout(open(log_path, 'w'))

    print(json.dumps(args.__dict__, indent = 4))

    print(json.dumps([model_config, training_config], indent = 4))

    ###  set the random seeds for deterministic results. ####
    SEED = args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    ### model preparation ###
    model = Transformer(**model_config)

    checkpoint_dir = os.getcwd() + f'/checkpoints/checkpoints_seed_{args.seed}/'
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_dir = os.path.join(checkpoint_dir, args.mode)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'{args.dataset}_{args.name}.model')
    training_config["checkpoint_path"] = checkpoint_path

    device_ids = list(range(torch.cuda.device_count()))
    #print(f"GPU list: {device_ids}")
    model = model.cuda()
    print(model)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("loading the best model from: " + checkpoint_path)

    ### data preparation ###

    train_ds = MyDataset(dataset=args.dataset, split='train', seed=args.seed, user_seq_len=args.user_seq_len, item_seq_len=args.item_seq_len)
    # dev_ds = MyDataset(dataset=args.dataset, split='valid', seed=args.seed, user_seq_len=args.user_seq_len, item_seq_len=args.item_seq_len)
    test_ds = MyDataset(dataset=args.dataset, split='test', seed=args.seed, user_seq_len=args.user_seq_len, item_seq_len=args.item_seq_len)

    ds_iter = {
            "train":DataLoader(train_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            # "dev":DataLoader(dev_ds, batch_size = training_config["batch_size"], shuffle=True, num_workers=4),
            "test":DataLoader(test_ds, batch_size = 1, shuffle=False, num_workers=4)
    }

    eval(model, ds_iter)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()