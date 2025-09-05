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
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

##############################################################################
# Loss / Metrics #
##############################################################################

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
    
    def BPR(self, output_batch, rating_batch):
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
    
    def NDCG(self, items, logits, ratings, k):
        items = torch.tensor(items)
        logits = torch.tensor(logits)
        ratings = torch.tensor(ratings)
        new_k = min((items!=0).sum().item(), k)
        if new_k==0:
            return None
        # ideal
        _,ideal_idx = torch.topk(ratings, new_k)
        ideal_items = torch.gather(items, -1, ideal_idx)
        ideal_ratings = torch.gather(ratings, -1, ideal_idx)
        # recommended
        _,rec_idx = torch.topk(logits, new_k)
        rec_items  = torch.gather(items,-1,rec_idx)
        rec_ratings = torch.gather(ratings,-1,rec_idx)
        # mask (ideal에 존재하는지 여부)
        rowA = rec_items.unsqueeze(1)
        rowB = ideal_items.unsqueeze(0)
        mask = (rowA==rowB).any(dim=1)
        rec_ratings *= mask
        # dcg/idcg/ndcg
        discount = torch.log2(torch.arange(new_k)+2)
        dcg = torch.sum(rec_ratings/discount, dim=-1)
        idcg = torch.sum(ideal_ratings/discount, dim=-1)
        ndcg = dcg/(idcg+1e-10).item()
        # 보정
        ndcg*=(new_k/k)
        return ndcg

    def rank_metrics(self, items, logits, ratings, k):
        items = torch.tensor(items)
        logits = torch.tensor(logits)
        ratings = torch.tensor(ratings)
        new_k = min((items!=0).sum().item(), k)
        if new_k==0:
            return None, None, None, None, None
        # ideal
        _,ideal_idx = torch.topk(ratings, new_k)
        ideal_items = torch.gather(items, -1, ideal_idx)
        ideal_ratings = torch.gather(ratings, -1, ideal_idx)
        # recommended
        _,rec_idx = torch.topk(logits, new_k)
        rec_items  = torch.gather(items,-1,rec_idx)
        rec_ratings = torch.gather(ratings,-1,rec_idx)
        # mask (ideal에 존재하는지 여부)
        rowA = rec_items.unsqueeze(1)
        rowB = ideal_items.unsqueeze(0)
        mask = (rowA==rowB).any(dim=1)
        rec_ratings *= mask
        # dcg/idcg/ndcg
        discount = torch.log2(torch.arange(new_k)+2)
        dcg = torch.sum(rec_ratings/discount, dim=-1)
        idcg = torch.sum(ideal_ratings/discount, dim=-1)
        ndcg = dcg/(idcg+1e-10).item()
        ndcg *= (new_k/k) # 보정
        # precision
        rec_items = rec_items.tolist()
        ideal_items = ideal_items.tolist()
        # print(set(rec_items))
        # print(set(items.tolist()))
        TP = set(rec_items).intersection(set(items.tolist()))
        precision = round(len(TP)/new_k, 4)
        recall = round(len(TP)/len(items.tolist()), 4)

        return pd.Series({
            f'ndcg@{k}':float(ndcg),
            f'precision@{k}':float(precision),
            f'recall@{k}':float(recall),
            f'rec@{k}':rec_items,
            f'ideal@{k}':ideal_items,
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