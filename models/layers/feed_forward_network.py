import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ffn_size, dropout):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(d_model, ffn_size)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(ffn_size, d_model)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x
    
class TopkRouter(nn.Module): # non-differentiable index를 방지하기위해 weight 결합 과정 필요
    def __init__(self, d_model, n_experts, topk, dropout):
        super().__init__()
        self.topk = topk
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self,x):
        logits = self.gate(x) # bs x l x n_experts
        # dev(gating logit에 noise추가)
        # noise = torch.randn_like(logits)
        # logits = logits+noise

        # load-balancing aux loss 계산용 : top-k로 마스킹되기 전, 전체 expert에 대한 밀집 분포
        dense_probs = F.softmax(logits, dim=-1) # bs x l x n_experts

        topk_logits, topk_indices = logits.topk(self.topk, dim=-1) # bs x l x topk
        zeros = torch.full_like(logits, float('-inf'), device=logits.device) # bs x l x n_experts
        sparse_logits = zeros.scatter(-1, topk_indices, topk_logits) # bs x l x n_experts
        output = F.softmax(sparse_logits, dim=-1) # bs x l x n_experts
        return output, topk_indices, dense_probs


class SparseMoE(nn.Module):
    def __init__(self, d_model, ffn_size, n_experts, topk, dropout):
        super(SparseMoE, self).__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.experts = nn.ModuleList([FeedForwardNetwork(d_model, ffn_size, dropout) for _ in range(n_experts)])
        self.router = TopkRouter(d_model=d_model, n_experts=n_experts, topk=topk, dropout=dropout)

        # 가장 최근 forward에서 계산된 load-balancing aux loss (계산만 하고 저장, 학습 loop에는 아직 미반영)
        self.last_aux_loss = None

    def compute_aux_loss(self, topk_indices, dense_probs):
        """
        Switch Transformer / GShard 스타일 load-balancing auxiliary loss.
            aux_loss = n_experts * sum_i (f_i * P_i)
                f_i: 실제로 top-k에 의해 expert i로 라우팅된 토큰 비율 (non-differentiable)
                P_i: expert i에 대한 평균 게이팅 확률(top-k 마스킹 전 전체 softmax 기준, differentiable)
        모든 expert가 균등하게 선택될 때 최솟값(=1)을 가지며, 특정 expert로의 쏠림(load imbalance)을 억제한다.
        아직 어떤 학습 loop에도 더해지지 않는, 계산만 해두는 상태.
        """
        assign_mask = torch.zeros(
            *topk_indices.shape[:-1], self.n_experts,
            device=topk_indices.device, dtype=dense_probs.dtype
        )
        assign_mask.scatter_(-1, topk_indices, 1.0) # bs x l x n_experts, 토큰별 선택된 expert 위치만 1

        f = assign_mask.mean(dim=tuple(range(assign_mask.dim() - 1))) # n_experts
        P = dense_probs.mean(dim=tuple(range(dense_probs.dim() - 1))) # n_experts

        return self.n_experts * torch.sum(f * P)

    def forward(self,x):
        # x : 128 x 30 x 64
        bs, l, d_model = x.shape
        gating_output, topk_indices, dense_probs = self.router(x) # bs x l x n_experts / bs x l x topk / bs x l x n_experts

        # aux loss는 계산해서 캐시만 해둠 (외부에서 필요하면 model.xxx.moe_ffn.last_aux_loss로 접근 가능)
        self.last_aux_loss = self.compute_aux_loss(topk_indices, dense_probs)

        final_output = torch.zeros_like(x) # bs x l x d_model

        # reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1)) # (bs*l) x d_model
        flat_gating_output = gating_output.view(-1, gating_output.size(-1)) # (bs*l) x n_experts
        flat_final = final_output.view(-1, self.d_model) # view -> writes propagate to final_output

        # process each expert. PERF: no `if flat_mask.any()` guard -- the `.any()` forces a
        # host<->device sync every expert every layer; an empty index_select/index_add is cheap.
        for i, expert in enumerate(self.experts):
            flat_mask = (topk_indices == i).any(dim=-1).view(-1) # (bs*l,)
            idx = flat_mask.nonzero(as_tuple=True)[0]
            expert_output = expert(flat_x[idx]) # k x d_model
            weighted_output = expert_output * flat_gating_output[idx, i].unsqueeze(1)
            flat_final.index_add_(0, idx, weighted_output)

        return final_output
