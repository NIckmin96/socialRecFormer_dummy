import torch
import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ffn_size, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(d_model, ffn_size)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(ffn_size, d_model)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        #x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x
    
class MoE(nn.Module):
    def __init__(self, d_model, ffn_size, n_experts=8, topk=1, dropout=0.1):
        super(MoE, self).__init__()
        self.topk = topk
        self.n_experts = n_experts
        self.softmax = nn.Softmax(dim=1)
        self.ffn_lst = nn.ModuleList()
        self.gate = nn.Linear(d_model, n_experts)
        for _ in range(n_experts):
            ffn = FeedForwardNetwork(d_model, ffn_size, dropout)
            self.ffn_lst.append(ffn)
        
        
    def forward(self,x):
        self.experts = self.gate(x) # bs x l x n_experts
        self.experts = self.softmax(self.experts) # bs x l x n_experts
        self.topk_weights, self.topk_indices = torch.topk(self.experts, self.topk, dim=-1)
        self.topk_indices = self.topk_indices.squeeze(-1) # bs x l
        bs, l, _ = x.shape
        output = []
        for b in range(bs):
            tmp = []
            for i in range(l):
                ffn = self.ffn_lst[self.topk_indices[b][i].item()]
                tmp.append(ffn(x[b][i]))
            output.append(torch.stack(tmp, dim=0))
        output = torch.stack(output, dim=0)
        
        return output