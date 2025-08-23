import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

class EncoderLayer(nn.Module):
    """
    Input:
        fixed-length random walk sequence (generated from social graph)
    """
    def __init__(self, d_model, d_ffn, num_heads, n_experts, topk, dropout):
        super(EncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x):
        # 1. Perform self attention
        residual = x
        x = self.norm1(x)
        x = self.attention(Q=x, K=x, V=x)
        # Add & Norm
        x = self.dropout1(x)
        x = x + residual

        # 3. FFN
        residual = x
        x = self.norm2(x)
        x = self.moe(x)
        # Add & Norm
        x = self.dropout2(x)
        x = x + residual

        return x