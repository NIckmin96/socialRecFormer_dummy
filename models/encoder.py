import torch
import torch.nn as nn

from models.blocks.encoder_layer import EncoderLayer, PredLayer
from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder

from model_utils import generate_attn_pad_mask

class Encoder(nn.Module):
    """
    Encoder for modeling user representation (in social graph)
    """
    # def __init__(self, max_degree, num_user, d_model, d_ffn, num_heads, dropout, num_layers):
    def __init__(self, user_seq_len, item_seq_len, min_item_len, user_embed, item_embed, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk):
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
                dropout = dropout
            ) for _ in range(num_layers)]
        )

        self.pred_layer = PredLayer(
            user_seq_len = user_seq_len,
            item_seq_len = min_item_len,
            d_model = d_model,
            d_ffn = d_ffn,
            num_heads = num_heads,
            n_experts = n_experts,
            topk = topk,
            dropout = dropout
        )
    
    def forward(self, batched_data):
        x = self.user_embed(batched_data['user_seq'], batched_data['user_degree'])
        x_item = self.item_embed(batched_data['item_list'], batched_data['item_degree'])
        # Generate mask for padded data
        user_mask = generate_attn_pad_mask(batched_data['user_seq'], batched_data['user_seq'])
        preference_mask = generate_attn_pad_mask(batched_data['user_seq'], batched_data['item_list'])
        # anchor user
        x_anchor_items = self.item_embed(batched_data['anchor_items'], batched_data['anchor_item_degree'])
        local_mask = generate_attn_pad_mask(batched_data['anchor_user'].unsqueeze(1), batched_data['anchor_items'])

        # Encoder layer forward pass (MHA, FFN)
        for layer in self.enc_layers:
            x, global_preference, x_item = layer(x, x_item, user_mask, preference_mask)

        self.global_attention = global_preference
        output = self.pred_layer(x, x_anchor_items, local_mask)
        
        # MF
        local_preference = torch.matmul(output.unsqueeze(1), x_anchor_items.transpose(2,1)).squeeze(1)
        
        return global_preference, local_preference
