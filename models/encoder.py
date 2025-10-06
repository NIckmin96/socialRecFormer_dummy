import torch
import torch.nn as nn

from models.blocks.encoder_layer import EncoderLayer
from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder

from model_utils import generate_attn_pad_mask

class Encoder(nn.Module):
    """
    Encoder for modeling user representation (in social graph)
    """
    # def __init__(self, max_degree, num_user, d_model, d_ffn, num_heads, dropout, num_layers):
    def __init__(self, user_seq_len, item_seq_len, user_embed, item_embed, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk, moe):
        """
        Args:
            data_path: path to dataset (ciao or epinions)
            spd_file: path to spd file (.npy)
            max_degree: max degree in social graph (can be fetched from `degree_table_social.csv`).
            num_user: number of total users in social graph (also can be fetched from `degree_table_social.csv`)
            d_model: embedding dimension (attention module)
            d_ffn: embedding dimension (FFN module)
            num_heads: number of heads in multi-headed attention
            dropout: dropout rate
            num_layers: number of encoder layers
        """
        super(Encoder, self).__init__()

        self.user_embed = user_embed
        self.item_embed = item_embed

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(
                user_seq_len = user_seq_len,
                item_seq_len = item_seq_len,
                d_model = d_model,
                d_ffn = d_ffn,
                num_heads = num_heads,
                n_experts = n_experts,
                topk = topk,
                dropout = dropout,
                moe = moe
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