import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

class DecoderLayer(nn.Module):
    
    def __init__(self, user_seq_len, item_seq_len, d_model, d_ffn, num_heads, n_experts=8, topk=1, dropout=0.1, last_layer:bool=False):
        super(DecoderLayer, self).__init__()

        self.last_layer_flag = last_layer

        # Self attention
        self.norm_self = nn.LayerNorm(d_model)
        # self.norm_self = nn.BatchNorm1d(item_seq_len)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout_self = nn.Dropout(p=dropout)
        # self attention - moe
        self.norm_self_moe = nn.LayerNorm(d_model)
        # self.norm_self_moe = nn.BatchNorm1d(item_seq_len)
        self.ffn_self = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.moe_self = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        self.dropout_self_moe = nn.Dropout(p=dropout)

        # Cross Attention(1) : user sequences - item sequences 간의 aggregation
        self.norm_cross1 = nn.LayerNorm(d_model)
        # self.norm_cross1 = nn.BatchNorm1d(item_seq_len)
        self.norm_cross1_enc = nn.LayerNorm(d_model)
        # self.norm_cross1_enc = nn.BatchNorm1d(user_seq_len)
        self.cross_attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout_cross1 = nn.Dropout(p=dropout)
        # Cross Attention(1) - moe
        self.norm_cross1_moe = nn.LayerNorm(d_model)
        # self.norm_cross1_moe = nn.BatchNorm1d(item_seq_len)
        self.dropout_cross1_moe = nn.Dropout(p=dropout)
        self.ffn_cross1 = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.moe_cross1 = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)


    def forward(self, x_item, enc_output, item_mask, item_user_mask):
        
        # 1. Self-Attention
        residual = x_item
        x_item = self.norm_self(x_item)
        x, _ = self.attention(Q=x_item, K=x_item, V=x_item, mask=item_mask)
        x = self.dropout_self(x)
        x = x + residual

        # 2-1. Cross Attention(1) : [anchor items - anchor user] 간의 attention
        residual = x
        x = self.norm_cross1(x)
        # enc_output = self.norm_cross1_enc(enc_output)[:,0,:].unsqueeze(1) # anchor user에 대한 representation만
        enc_output = self.norm_cross1_enc(enc_output)
        x, attention = self.cross_attention1(Q=x, K=enc_output, V=enc_output, mask=item_user_mask) # bs x 1 x d
        x = self.dropout_cross1(x)
        x = x + residual 
        
        # 2-2. MoE / FFN
        residual = x
        x = self.norm_cross1_moe(x)
        # x = self.ffn_cross1(x)
        x = self.moe_cross1(x)
        x = self.dropout_cross1_moe(x)
        x = x + residual
        
        return x, attention, enc_output