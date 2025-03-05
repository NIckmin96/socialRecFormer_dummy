import torch.nn as nn

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
        self.moe = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)

        # Self attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_dec_layer=self.dec_layer)
        self.dropout1 = nn.Dropout(p=dropout)
        self.moe_norm_self = nn.LayerNorm(d_model)

        # Cross Attention(1) : user sequences - item sequences 간의 aggregation
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_dec_layer=self.dec_layer, is_rating=True)
        self.dropout2 = nn.Dropout(p=dropout)
        
        # last layer가 아닌 경우에는 FFN 거침
        if not self.last_layer_flag:
            
            # Cross Attention(1)의 FFN layer
            self.moe_norm_cross1 = nn.LayerNorm(d_model)
            # Cross Attention(2) : anchor user - item sequences 간의 aggregation
            self.norm3 = nn.LayerNorm(d_model)
            self.cross_attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, is_dec_layer=self.dec_layer, is_rating=False)
            self.dropout3 = nn.Dropout(p=dropout)
            self.moe_norm_cross2 = nn.LayerNorm(d_model)
        
        if self.last_layer_flag:
            # prediction layer
            self.last_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, last_layer_flag=True, is_dec_layer=self.dec_layer, is_rating=True)

    def forward(self, x_item, x_anchor, x_anchor_i, rating_x, enc_output, self_attn_mask, cross_attn_mask_1, cross_attn_mask_2, rating_bias, ranking_bias):
        # tmp
        bce_loss, rmse_loss = 0,0
        
        # 1-1. Self-Attention
        residual = x_item
        x, _ = self.attention(Q=x_item, K=x_item, V=x_item, mask=self_attn_mask, attn_bias=None)
        # Add & Norm
        x = x + residual
        x = self.norm1(x)
        
        # 1-2. FFN
        residual = x
        x = self.moe(x)
        # Add & Norm
        x = x + residual
        x = self.moe_norm_self(x)
        x = self.dropout1(x)
        
        # 2-1. Cross Attention(1) : [user sequences - item sequences] 간의 aggregation
        residual = x
        enc_output = enc_output + rating_x # rating 정보 추가
        
        if not self.last_layer_flag:
            x, rmse_loss = self.cross_attention1(Q=x, K=enc_output, V=enc_output, mask=cross_attn_mask_1, attn_bias=rating_bias)
            # Add & Norm
            x = x + residual
            x = self.norm2(x)
        
            # 2-2. FFN
            residual = x
            x = self.moe(x)
            # Add & Norm
            x = x + residual
            x = self.moe_norm_cross1(x)
            x = self.dropout2(x)
            
            # 3-1. Cross Attention(2) : (anchor user - item sequences) + (user-item seq representation)의 aggregation
            residual = x
            x = x + x_anchor_i # user-item representation + anchor user기준 item embeddings
            new_enc_output = enc_output + x_anchor.expand(-1,enc_output.size(1),-1)
            x, bce_loss = self.cross_attention2(Q=x, K=x_anchor, V=x_anchor, mask=cross_attn_mask_2, attn_bias=ranking_bias)
            # x, _ = self.cross_attention2(Q=x, K=new_enc_output, V=new_enc_output, mask=cross_attn_mask_2, attn_bias=ranking_bias)
            # Add & Norm
            x = x + residual
            x = self.norm3(x)
        
            # 3-2. FFN
            residual = x
            x = self.moe(x)
            # Add & Norm
            x = x + residual
            x = self.moe_norm_cross2(x)
            x = self.dropout3(x)
            
            return x, bce_loss, rmse_loss

        else:
            # 7. last layer returns predicted ratings.
            x, rmse_loss = self.last_attn(Q=x, K=enc_output, V=enc_output, mask=cross_attn_mask_1, attn_bias=rating_bias)
            return x, bce_loss, rmse_loss