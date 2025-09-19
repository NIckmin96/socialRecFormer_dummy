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
    def __init__(self, user_seq_len, item_seq_len, d_model, d_ffn, num_heads, n_experts=8, topk=1, dropout=0.1, last_layer:bool=False):
        super(DecoderLayer, self).__init__()

        self.last_layer_flag = last_layer

        # Self attention
        # self.norm_self = nn.LayerNorm(d_model)
        self.norm_self = nn.BatchNorm1d(item_seq_len)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout_self = nn.Dropout(p=dropout)
        # self attention - moe
        # self.norm_self_moe = nn.LayerNorm(d_model)
        self.norm_self_moe = nn.BatchNorm1d(item_seq_len)
        self.ffn_self = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.moe_self = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        self.dropout_self_moe = nn.Dropout(p=dropout)

        # Cross Attention(1) : user sequences - item sequences 간의 aggregation
        # self.norm_cross1 = nn.LayerNorm(d_model)
        self.norm_cross1 = nn.BatchNorm1d(item_seq_len)
        # self.norm_cross1_enc = nn.LayerNorm(d_model)
        self.norm_cross1_enc = nn.BatchNorm1d(user_seq_len)
        self.cross_attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout_cross1 = nn.Dropout(p=dropout)
        # Cross Attention(1) - moe
        # self.norm_cross1_moe = nn.LayerNorm(d_model)
        self.norm_cross1_moe = nn.BatchNorm1d(item_seq_len)
        self.dropout_cross1_moe = nn.Dropout(p=dropout)
        self.ffn_cross1 = FeedForwardNetwork(d_model, d_ffn, dropout)
        self.moe_cross1 = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)

        # # prediction layer
        # self.norm_last = nn.LayerNorm(d_model)
        # self.norm_last_enc = nn.LayerNorm(d_model)
        # self.last_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, last_layer_flag=True)
        # self.last_activation = nn.LeakyReLU()
        
        # # Cross Attention(2) : anchor user - item sequences 간의 aggregation
        # self.x_lin = nn.Linear(d_model, d_model//2)
        # self.anchor_i_lin = nn.Linear(d_model, d_model//2)
        # self.enc_lin = nn.Linear(d_model, d_model//2)
        # self.anchor_u_lin = nn.Linear(d_model, d_model//2)

        # self.norm_cross2 = nn.LayerNorm(d_model)
        # self.norm_cross2_enc = nn.LayerNorm(d_model)
        
        # self.cross_attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        # self.dropout_cross2 = nn.Dropout(p=dropout)
        # # Cross Attention(2) : ffn
        # self.norm_cross2_moe = nn.LayerNorm(d_model)
        # self.ffn_cross2 = FeedForwardNetwork(d_model, d_ffn, dropout)
        # self.moe_cross2 = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        # self.dropout_cross2_moe = nn.Dropout(p=dropout)

    def forward(self, x_item, enc_output, item_mask, item_user_mask):
        
        # 1. Self-Attention
        residual = x_item
        x_item = self.norm_self(x_item)
        x, _ = self.attention(Q=x_item, K=x_item, V=x_item, mask=item_mask)
        x = self.dropout_self(x)
        x = x + residual
        
        # # 1-1. MoE
        # residual = x
        # x = self.norm_self_moe(x)
        # # x = self.ffn_self(x)
        # x = self.moe_self(x)
        # x = self.dropout_self_moe(x)
        # x = x + residual

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
        
        # if self.last_layer_flag:
        #     x = self.norm_last(x)
        #     enc_output = self.norm_last_enc(enc_output)
        #     # last layer에서 attention하지 않고, representation간의 matmul을 통해 MF
        #     rating_pred = torch.matmul(x, enc_output.transpose(2,1))
        #     # x, rmse_loss, rating_pred = self.last_attn(Q=x, K=enc_output, V=enc_output, mask=cross_attn_mask_1, attn_bias=None)
            
        #     rating_pred = self.last_activation(rating_pred)
        
        # else:
        #     # 2-2. FFN
        #     residual = x
        #     x = self.norm_cross1_moe(x)
        #     # x = self.ffn_cross1(x)
        #     x = self.moe_cross1(x)
        #     x = self.dropout_cross1_moe(x)
        #     x = x + residual

        # # 3-1. Cross Attention(2) : (anchor user - item sequences) + (user-item seq representation)의 aggregation
        # residual = x
        # ######################## concat ########################
        # # x = self.x_lin(x)
        # # x_anchor_i = self.anchor_i_lin(x_anchor_i)
        # # x = torch.cat((x, x_anchor_i), dim=-1)

        # # x_anchor_u = self.anchor_u_lin(x_anchor)
        # # enc_output = self.enc_lin(enc_output)
        # # new_enc_output = torch.cat((enc_output, x_anchor_u.expand(*enc_output.size())), dim=-1)
        # ######################## addition ######################
        # x = x+x_anchor_i
        # new_enc_output = enc_output+x_anchor.expand(*enc_output.size())
        # ########################################################

        # x = self.norm_cross2(x)
        # new_enc_output = self.norm_cross2_enc(new_enc_output)
        
        # x, _ = self.cross_attention2(Q=x, K=new_enc_output, V=new_enc_output, mask=cross_attn_mask_2, attn_bias=None)
        # x = self.dropout_cross2(x)
        # x = x + residual
    
        # # 3-2. FFN
        # residual = x
        # x = self.norm_cross2_moe(x)
        # # x = self.ffn_cross2(x)
        # x = self.moe_cross2(x)
        # x = self.dropout_cross2_moe(x)
        # x = x + residual
        
        return x, attention