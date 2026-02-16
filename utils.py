import sys
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

##############################################################################
# weight initialization #
##############################################################################
def he_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

##############################################################################
# lr scheduler #
##############################################################################
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

##############################################################################
# Loss / Metrics #
##############################################################################

def filter_negs(x):
    mask = (torch.tensor(x['anchor_ratings'])!=0)
    items = torch.tensor(x['anchor_items'])
    items = items[mask].tolist()
    outputs = torch.tensor(x['outputs'])[mask].tolist()
    ratings = torch.tensor(x['anchor_ratings'])[mask].tolist()
    
    return pd.Series({'anchor_user':x['anchor_user'], 'anchor_items':items, 'outputs':outputs,'anchor_ratings':ratings})

class Metrics:
    def MAE(self, pred, y, mask=None):
        if mask==None:
            mask = torch.ones_like(pred).long()
        
        return F.l1_loss(pred[mask].float(), y[mask].float(), reduction='mean')

    def MSE(self, pred, y, mask=None):
        if mask==None:
            mask = torch.ones_like(pred).long()
        
        return F.mse_loss(pred[mask].float(), y[mask].float(), reduction='mean')

    def RMSE(self, pred, y, mask=None):
        if mask==None:
            mask = torch.ones_like(pred).long()

        return torch.sqrt(self.MSE(pred, y, mask))
    
    def BPR(self, output_batch, rating_batch, neg):
        bpr_loss = 0.0
        
        if neg:
            logits = output_batch[rating_batch==0]
            logit_0 = logits.sum() if logits.numel()==0 else logits.mean()
        
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
                diff = pos-(neg+0.1)
                loss = torch.log(1 + torch.abs(diff - 1))
                bpr_loss += loss
    
        else:
            for neg,pos in [(logit_1, logit_2), (logit_2, logit_3), (logit_3, logit_4), (logit_4, logit_5)]:
                diff = pos-(neg+0.1)
                loss = torch.log(1 + torch.abs(diff - 1))
                bpr_loss += loss
            
        return bpr_loss
    
    def NDCG(self, items, logits, ratings, k):
        items = torch.tensor(items)
        logits = torch.tensor(logits)
        ratings = torch.tensor(ratings)
        new_k = min(len(items.tolist()), k)
        if new_k==0:
            return None
        # ideal
        _,ideal_idx = torch.topk(ratings, new_k)
        ideal_items = items[ideal_idx]
        ideal_ratings = ratings[ideal_idx] 
        # recommended
        _,rec_idx = torch.topk(logits, new_k)
        rec_items = items[rec_idx]
        rec_ratings = ratings[rec_idx]
        # dcg/idcg/ndcg
        discount = torch.log2(torch.arange(new_k)+2)
        dcg = torch.sum(rec_ratings/discount, dim=-1)
        idcg = torch.sum(ideal_ratings/discount, dim=-1)
        ndcg = dcg/(idcg+1e-10).item()
        return ndcg
    
    # def rank_metrics(self, items, logits, ratings, k=10):
    #     eps = 1e-10
    #     new_k = min(k, len(ratings))
    #     gt_items = items[ratings!=0] # 실제로 interact한 items
    #     _, rec_indices = torch.from_numpy(logits).topk(new_k)
    #     recommended_i = torch.from_numpy(items)[rec_indices].flatten()
    #     recommended_r = torch.from_numpy(ratings)[rec_indices].flatten()
        
    #     _, ideal_indices = torch.from_numpy(ratings).topk(new_k)
    #     ideal_i = torch.from_numpy(items)[ideal_indices].flatten()
    #     ideal_r = torch.from_numpy(ratings)[ideal_indices].flatten()
    #     discount = torch.log2(torch.arange(new_k)+2)
    #     item_mask = torch.tensor(list(map(lambda x:1 if x in set(gt_items.tolist()) else 0, list(recommended_i))))
        
    #     DCG = torch.sum(recommended_r*item_mask/discount)
    #     IDCG = torch.sum(ideal_r/discount)
    #     NDCG = DCG/(IDCG+eps)
    #     assert NDCG<=1.0
        
    #     # precision
    #     TP = set(recommended_i.tolist()).intersection(set(gt_items.tolist()))
    #     precision = round(len(TP)/new_k, 4) if new_k>0 else 0.0
    #     recall = round(len(TP)/len(gt_items), 4) if len(gt_items)>0 else 0.0
        
    #     return NDCG, new_k, precision, recall
        
    def rank_metrics(self, x, k):
        user = x['anchor_user']
        items = torch.tensor(x['anchor_items'])
        ratings = torch.tensor(x['anchor_ratings'])
        logits = torch.tensor(x['logits'])
        new_k = min(len(items.tolist()), k)
        if new_k==0:
            return None, None, None, None, None
        # ideal
        _,ideal_idx = torch.topk(ratings, new_k)
        ideal_items = items[ideal_idx]
        ideal_ratings = ratings[ideal_idx]
        # ideal_items = torch.gather(items, -1, ideal_idx)
        # ideal_ratings = torch.gather(ratings, -1, ideal_idx)
        # recommended
        _,rec_idx = torch.topk(logits, new_k)
        rec_items = items[rec_idx]
        rec_ratings = ratings[rec_idx]
        # rec_items  = torch.gather(items,-1,rec_idx)
        # rec_ratings = torch.gather(ratings,-1,rec_idx)
        
        # # mask (ideal에 존재하는지 여부)
        # rowA = rec_items.unsqueeze(1)
        # rowB = ideal_items.unsqueeze(0)
        # mask = (rowA==rowB).any(dim=1)
        # rec_ratings *= mask
        # dcg/idcg/ndcg
        discount = torch.log2(torch.arange(new_k)+2)
        dcg = torch.sum(rec_ratings/discount, dim=-1)
        idcg = torch.sum(ideal_ratings/discount, dim=-1)
        ndcg = dcg/(idcg+1e-10).item()
        # ndcg *= (new_k/k) # 보정
        # precision / recall
        rec_items = rec_items.tolist()
        ideal_items = ideal_items.tolist()
        gt_items = items[ratings!=0].tolist()
        TP = set(rec_items).intersection(set(gt_items))
        hr = 1 if len(TP)>=1 else 0
        precision = round(len(TP)/new_k, 4)
        recall = round(len(TP)/len(gt_items), 4)
        # HR@k
        

        return pd.Series({
            'user':user,
            'items':items.tolist(),
            'ratings':ratings.tolist(),
            'logits':logits.tolist(),
            f'ndcg@{k}':float(ndcg),
            f'precision@{k}':float(precision),
            f'recall@{k}':float(recall),
            f'rec@{k}':rec_items,
            f'ideal@{k}':ideal_items,
            f'hr@{k}':hr,
        })
        
        # return ndcg, precision, recall, rec_items, ideal_items

##############################################################################
# REDIRECT LOGGER #
##############################################################################

def redirect_stdout(logfile):
    def MyHookOut(text):
        logfile.write(text)
        logfile.flush()
        return 1, 0, text
    phOut = PrintHook()
    phOut.Start(MyHookOut)


# this class gets all output directed to stdout(e.g by print statements)
# and stderr and redirects it to a user defined function
class PrintHook:
    # out = 1 means stdout will be hooked
    # out = 0 means stderr will be hooked
    def __init__(self, out=1):
        self.func = None  ##self.func is userdefined function
        self.origOut = None
        self.out = out

    # user defined hook must return three variables
    # proceed, lineNoMode, newText
    def TestHook(self, text):
        f = open('hook_log.txt', 'a')
        f.write(text)
        f.close()
        return 0, 0, text

    def Start(self, func=None):
        if self.out:
            sys.stdout = self
            self.origOut = sys.__stdout__
        else:
            sys.stderr = self
            self.origOut = sys.__stderr__

        if func:
            self.func = func
        else:
            self.func = self.TestHook

    # Stop will stop routing of print statements thru this class
    def Stop(self):
        self.origOut.flush()
        if self.out:
            sys.stdout = sys.__stdout__
        else:
            sys.stderr = sys.__stderr__
        self.func = None

    # override write of stdout
    def write(self, text):

        bProceed = 1
        bLineNo = 0
        newText = ''

        if self.func != None:
            bProceed, bLineNo, newText = self.func(text)

        if bProceed:
            if text.split() == []:
                self.origOut.write(text)
            else:
                # if goint to stdout then only add line no file etc
                # for stderr it is already there
                if self.out:
                    if bLineNo:
                        try:
                            raise Exception("Dummy")
                        except:
                            lineNo = 'line(' + str(sys.exc_info()[2].tb_frame.f_back.f_lineno) + '):'
                            codeObject = sys.exc_info()[2].tb_frame.f_back.f_code
                            fileName = codeObject.co_filename
                            funcName = codeObject.co_name
                        self.origOut.write('file ' + fileName + ',' + 'func ' + funcName + ':' + lineNo)
                self.origOut.write(newText)

    # pass all other methods to __stdout__ so that we don't have to override them
    def __getattr__(self, name):
        # return self.origOut.__getattr__(name)
        return getattr(self.origOut, name)


if __name__ == '__main__':

    test_printhook = False
    if test_printhook:

        def MyHookOut(text):
            f = open('log.txt', 'a')
            f.write(text)
            f.close()
            return 1, 1, 'Out Hooked:' + text


        def MyHookErr(text):
            f = open('hook_log.txt', 'a')
            f.write(text)
            f.close()
            return 1, 1, 'Err Hooked:' + text


        print('Hook Start')
        phOut = PrintHook()
        phOut.Start(MyHookOut)
        phErr = PrintHook(0)
        phErr.Start(MyHookErr)
        print('Is this working?')
        print('It seems so!')
        phOut.Stop()
        print('STDOUT Hook end')
        compile(',', '<string>', 'exec')
        phErr.Stop()
        print('Hook end')
