import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

class EncoderLayer(nn.Module):
    """
    Input:
        fixed-length random walk sequence (generated from social graph)
    """
    def __init__(self, user_seq_len, item_seq_len, d_model, d_ffn, num_heads, n_experts=8, topk=1, dropout=0.1):
        super(EncoderLayer, self).__init__()

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm1 = nn.BatchNorm1d(user_seq_len)
        self.attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)

        # self.norm_moe1 = nn.LayerNorm(d_model)
        self.norm_moe1 = nn.BatchNorm1d(user_seq_len)
        self.moe1 = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        self.ffn1 = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.dropout_moe1 = nn.Dropout(p=dropout)
        
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm2 = nn.BatchNorm1d(user_seq_len)
        # self.norm2_item = nn.LayerNorm(d_model)
        self.norm2_item = nn.BatchNorm1d(item_seq_len)
        self.attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout)

        # self.norm_moe2 = nn.LayerNorm(d_model)
        self.norm_moe2 = nn.BatchNorm1d(user_seq_len)
        self.moe2 = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        self.ffn2 = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.dropout_moe2 = nn.Dropout(p=dropout)
    
    def forward(self, x, x_item, user_mask, preference_mask):
        # 1. Perform self attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention1(Q=x, K=x, V=x, mask=user_mask)
        # Add & Norm
        x = self.dropout1(x)
        x = x + residual

        # # 1-1. FFN
        # residual = x
        # x = self.norm_moe1(x)
        # # x = self.ffn1(x)
        # x = self.moe1(x)
        # # Add & Norm
        # x = self.dropout_moe1(x)
        # x = x + residual
        
        # 2. user-item attention(여기서 global rating prediction ouptut으로 뽑을수있게)
        residual = x
        x = self.norm2(x)
        x_item = self.norm2_item(x_item)
        x, attention = self.attention2(Q=x, K=x_item, V=x_item, mask=preference_mask)
        x = self.dropout2(x)
        x = x + residual
        
        # 2-1. MoE/FFN
        residual = x
        x = self.norm_moe2(x)
        # x = self.ffn2(x)
        x = self.moe2(x)
        # Add & Norm
        x = self.dropout_moe2(x)
        x = x + residual

        return x, attention