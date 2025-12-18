import torch
import torch.nn as nn

from models.blocks.encoder_layer import EncoderLayer
from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

from model_utils import generate_attn_pad_mask

class Encoder(nn.Module):
    def __init__(self, user_seq_len, item_seq_len, user_embed, item_embed, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk, moe):
        super(Encoder, self).__init__()
        
        if moe:
            moe_ffn = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        else:
            moe_ffn = FeedForwardNetwork(d_model, d_ffn, dropout)

        self.user_embed = user_embed
        self.item_embed = item_embed

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(
                user_seq_len = user_seq_len,
                item_seq_len = item_seq_len,
                d_model = d_model,
                num_heads = num_heads,
                dropout = dropout,
                moe = moe_ffn
            ) for _ in range(num_layers)]
        )
    
    def forward(self, batched_data):
        x = self.user_embed(batched_data['user_seq'], batched_data['user_degree'])
        x_item = self.item_embed(batched_data['item_list'], batched_data['item_degree'])
        # Generate mask for padded data
        self.user_mask = generate_attn_pad_mask(batched_data['user_seq'], batched_data['user_seq'])
        self.preference_mask = generate_attn_pad_mask(batched_data['user_seq'], batched_data['item_list'])

        # Encoder layer forward pass (MHA, FFN)
        for layer in self.enc_layers:
            x, attention, x_item = layer(x, x_item, self.user_mask, self.preference_mask)
            
        self.global_attention = attention
        # MF
        global_preference = torch.matmul(x, x_item.transpose(2,1))
        
        return x, global_preference