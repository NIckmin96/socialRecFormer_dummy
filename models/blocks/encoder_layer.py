import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    """
    Input:
        fixed-length random walk sequence (generated from social graph)
    """
    def __init__(self, user_seq_len, item_seq_len, d_model, num_heads, dropout, moe):
        super(EncoderLayer, self).__init__()
        # self.moe = moe # T/F
        self.moe_ffn = moe # MoE/FFN 모듈 자체를 받음

        # self.self_norm = nn.BatchNorm1d(user_seq_len)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.self_dropout = nn.Dropout(p=dropout)

        # self.norm_moe1 = nn.BatchNorm1d(user_seq_len)
        self.norm_moe1 = nn.LayerNorm(d_model)
        self.dropout_moe1 = nn.Dropout(p=dropout)
        
        # self.prefer_norm = nn.BatchNorm1d(user_seq_len)
        self.prefer_norm = nn.LayerNorm(d_model)
        # self.prefer_norm_item = nn.BatchNorm1d(item_seq_len)
        self.prefer_norm_item = nn.LayerNorm(d_model)
        self.prefer_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.prefer_dropout = nn.Dropout(p=dropout)

        # self.norm_moe2 = nn.BatchNorm1d(user_seq_len)
        self.norm_moe2 = nn.LayerNorm(d_model)
        self.dropout_moe2 = nn.Dropout(p=dropout)
    
    def forward(self, x, x_item, user_mask, preference_mask):
        # 1. Perform self attention
        residual = x
        x = self.self_norm(x)
        x, _ = self.self_attention(Q=x, K=x, V=x, mask=user_mask)
        # Add & Norm
        x = self.self_dropout(x)
        x = x + residual
        
        # 2. user-item attention(여기서 global rating prediction ouptut으로 뽑을수있게)
        residual = x
        x = self.prefer_norm(x)
        # x_item은 다음 블록에도 원본 그대로 전달되어야 하므로, 정규화는 지역 변수에만 적용
        x_item_normed = self.prefer_norm_item(x_item)
        x, attention = self.prefer_attention(Q=x, K=x_item_normed, V=x_item_normed, mask=preference_mask)
        x = self.prefer_dropout(x)
        x = x + residual
        
        # 2-1. MoE/FFN
        residual = x
        x = self.norm_moe2(x)            
        x = self.moe_ffn(x)
        
        # Add & Norm
        x = self.dropout_moe2(x)
        x = x + residual

        return x, attention, x_item