import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class TopkRouter(nn.Module): # non-differentiable index를 방지하기위해 weight 결합 과정 필요
    def __init__(self, d_model, n_experts=8, topk=1):
        super().__init__()
        self.topk = topk
        self.gate = nn.Linear(d_model, n_experts)
    
    def forward(self,x):
        logits = self.gate(x) # bs x l x n_experts
        topk_logits, topk_indices = logits.topk(self.topk, dim=-1) # bs x l x topk
        zeros = torch.full_like(logits, float('-inf'), device=logits.device) # bs x l x n_experts
        sparse_logits = zeros.scatter(-1, topk_indices, topk_logits) # bs x l x n_experts
        output = F.softmax(sparse_logits, dim=-1) # bs x l x n_experts
        return output, topk_indices
    
    
class SparseMoE(nn.Module):
    def __init__(self, d_model, ffn_size, n_experts=8, topk=1, dropout=0.1):
        super(SparseMoE, self).__init__()
        self.d_model = d_model
        self.experts = nn.ModuleList([FeedForwardNetwork(d_model, ffn_size) for _ in range(n_experts)])
        self.router = TopkRouter(d_model=d_model, n_experts=n_experts, topk=topk)
        
        
    def forward(self,x):
        # x : 128 x 30 x 64
        bs, l, d_model = x.shape
        gating_output, topk_indices = self.router(x) # bs x l x n_experts / bs x l x topk
        final_output = torch.zeros((bs, l, self.d_model), device=x.device) # (bs * l) x d_model
        
        # reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1)) # (bs*l) x d_model
        flat_gating_output = gating_output.view(-1, gating_output.size(-1)) # (bs*l) x n_experts
        
        # process each expert in parallel
        for i,expert in enumerate(self.experts):
            expert_mask = (topk_indices==i).any(dim=-1) # bs x l
            flat_mask = expert_mask.view(-1) # (bs*l)
            if flat_mask.any():
                expert_input = flat_x[flat_mask] # (bs*l) x d_model
                expert_output = expert(expert_input) # (bs*l) x d_model
                
                # extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores
                
                final_output[expert_mask] += weighted_output.squeeze(1)
        
        return final_output