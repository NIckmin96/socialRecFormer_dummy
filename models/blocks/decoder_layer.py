import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

class DecoderLayer(nn.Module):
    """
    Input:
        fixed-length item sequences \n
        This items are interacted items of users in encoder's input random walk sequence.
    """
    def __init__(self, d_model, d_ffn, num_heads, n_experts=8, topk=1, dropout=0.1, last_layer:bool=False, is_dec_layer:bool=True):
        super(DecoderLayer, self).__init__()

        self.last_layer_flag = last_layer
        self.dec_layer = is_dec_layer
        
        # FFN(Sparse MoE)
        # self.moe = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)

        # Self attention
        self.norm_self = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_dec_layer=self.dec_layer)
        self.dropout_self = nn.Dropout(p=dropout)
        # self attention - ffn
        self.norm_self_ffn = nn.LayerNorm(d_model)
        self.ffn_self = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.dropout_self_ffn = nn.Dropout(p=dropout)

        # Cross Attention(1) : user sequences - item sequences 간의 aggregation
        self.norm_cross1 = nn.LayerNorm(d_model)
        self.cross_attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_dec_layer=self.dec_layer, is_rating=True)
        self.dropout_cross1 = nn.Dropout(p=dropout)
        # Cross Attention(1) - ffn
        self.norm_cross1_ffn = nn.LayerNorm(d_model)
        self.ffn_cross_1 = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.dropout_cross1_ffn = nn.Dropout(p=dropout)
        
            
        # Cross Attention(2) : anchor user - item sequences 간의 aggregation
        self.norm_cross2 = nn.LayerNorm(d_model)
        self.cross_attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_dec_layer=self.dec_layer, is_rating=False)
        self.dropout_cross2 = nn.Dropout(p=dropout)
        # Cross Attention(2) : ffn
        self.norm_cross2_ffn = nn.LayerNorm(d_model)
        self.ffn_cross_2 = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.dropout_cross2_ffn = nn.Dropout(p=dropout)
        
        # prediction layer
        self.last_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, last_layer_flag=True, is_dec_layer=self.dec_layer, is_rating=True)
        self.activation = nn.LeakyReLU(d_model)

    def forward(self, x_item, x_anchor, x_anchor_i, rating_x, enc_output, self_attn_mask, cross_attn_mask_1, cross_attn_mask_2, rating_bias, ranking_bias):
        # tmp
        rmse_loss = 0
        rating_pred = None
        
        # 1. Self-Attention
        residual = x_item
        x_item = self.norm_self(x_item)
        x, _ = self.attention(Q=x_item, K=x_item, V=x_item, mask=self_attn_mask, attn_bias=None)
        x = self.dropout_self(x)
        x = x + residual
        # 1-1. FFN
        residual = x
        x = self.norm_self_ffn(x)
        x = self.ffn_self(x)
        x = self.dropout_self_ffn(x)
        x = x + residual

        # 2-1. Cross Attention(1) : [user sequences - item sequences] 간의 aggregation
        residual = x
        enc_output = enc_output # rating 정보 추가 x
        # enc_output = enc_output + rating_x # rating 정보 추가
        x = self.norm_cross1(x)
        x, rmse_loss = self.cross_attention1(Q=x, K=enc_output, V=enc_output, mask=cross_attn_mask_1, attn_bias=rating_bias)
        x = self.dropout_cross1(x)
        x = x + residual
        
        if self.last_layer_flag:
            x, rmse_loss, rating_pred = self.last_attn(Q=x, K=enc_output, V=enc_output, mask=cross_attn_mask_1, attn_bias=rating_bias)
            
        # 2-2. FFN
        residual = x
        x = self.norm_cross1_ffn(x)
        x = self.ffn_cross_1(x)
        x = self.dropout_cross1_ffn(x)
        x = x + residual
        
        # 3-1. Cross Attention(2) : (anchor user - item sequences) + (user-item seq representation)의 aggregation
        residual = x
        x = x + x_anchor_i # [TODO] Concatenation으로 변경해서 실험
        x = self.norm_cross2(x)
        new_enc_output = enc_output + x_anchor.expand(-1,enc_output.size(1),-1) # [TODO] Concatenation으로 변경해서 실험
        x, _ = self.cross_attention2(Q=x, K=new_enc_output, V=new_enc_output, mask=cross_attn_mask_2, attn_bias=None)
        # Ranking loss(BCE)
        x = self.dropout_cross2(x)
        x = x + residual
    
        # 3-2. FFN
        residual = x
        x = self.norm_cross2_ffn(x)
        x = self.ffn_cross_2(x)
        x = self.dropout_cross2_ffn(x)
        x = x + residual

        # 3-3. activation
        x = self.activation(x)
        
        return x, rmse_loss, rating_pred