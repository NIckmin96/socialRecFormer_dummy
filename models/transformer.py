import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import CE
from models.encoder import Encoder
from models.layers.encoding_modules import Embedding

class Transformer(nn.Module):
    # def __init__(self, num_user, max_degree_user, num_item, max_degree_item, d_model, d_ffn, num_heads, dropout, num_layers_enc, num_layers_dec):
    def __init__(self, seq_len, max_node, max_degree, d_model, d_ffn, num_heads, dropout, num_layers_enc, n_experts, topk):
        super(Transformer, self).__init__()
        
        self.input_embed = Embedding(
            max_node = max_node,
            max_degree = max_degree,
            d_model = d_model
        )

        self.encoder = Encoder(
            input_embed=self.input_embed,
            d_model=d_model,
            d_ffn=d_ffn,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers_enc,
            n_experts=n_experts,
            topk=topk
        )
        
        self.x_linear = nn.Linear(2*d_model, d_model)
        self.linear = nn.Linear(seq_len, 1)
    
    def forward(self, batch1, batch2):
        attn_output = self.encoder(batch1) # bs x n x d
        mask = (batch2['product']!=0) # bs x i
        # print(batch2['user'].max(), batch2['user_degree'].max())
        item_embed = self.input_embed(batch2['product'], batch2['product_degree']) # bs x i x d
        user_embed = self.input_embed(batch2['user'], batch2['user_degree']).expand(*item_embed.size()) # bs x i x d
        x = torch.cat([user_embed, item_embed], dim=-1)
        x = self.x_linear(x) # bs x i x d
        x = torch.matmul(x, attn_output.transpose(-1,-2)) # bs x i x n
        x = self.linear(x).squeeze(-1) # bs x i
        # CE loss
        target = F.softmax(batch2['ratings'].float(), dim=-1)*mask # bs x i
        logits = F.softmax(x, dim=-1)*mask
        loss = CE(logits, target)
        
        return loss, target, logits