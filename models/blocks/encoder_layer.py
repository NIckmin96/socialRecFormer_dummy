import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork

class EncoderLayer(nn.Module):
    """
    Input:
        fixed-length random walk sequence (generated from social graph)
    """
    def __init__(self, d_model, d_ffn, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model=d_model, ffn_size=d_ffn, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x, src_mask, attn_bias):
        # 1. Perform self attention
        residual = x
        x = self.norm1(x)
        x, spd_loss = self.attention(Q=x, K=x, V=x, mask=src_mask, attn_bias=attn_bias)

        # 2. Add & Norm
        x = self.dropout1(x)
        x = x + residual

        # 3. FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)

        # 4. Add & Norm
        x = self.dropout2(x)
        x = x + residual

        return x, spd_loss