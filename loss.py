import torch
import torch.nn.functional as F
from torch import nn

class BPRLoss(nn.Module):
    def __init__(self, lamb_reg):
        super(BPRLoss, self).__init__()
        self.lamb_reg = lamb_reg

    def forward(self, pos_preds, neg_preds, *reg_vars):
        batch_size = pos_preds.size(0)
        pos_scores = torch.sum(pos_preds, dim=-1)
        neg_scores = torch.sum(neg_preds, dim=-1)

        # Compute the BPR loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        #bpr_loss = -0.5 * (pos_preds - neg_preds).sigmoid().log().sum() / batch_size

        reg_loss = torch.tensor([0.], device=bpr_loss.device)
        for var in reg_vars:
            reg_loss += self.lamb_reg * 0.5 * var.pow(2).sum()
        reg_loss /= batch_size

        loss = bpr_loss + reg_loss

        return loss, [bpr_loss.item(), reg_loss.item()]
    