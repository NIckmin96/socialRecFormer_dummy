import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder

from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder, ItemNodeEncoder, RatingEncoder, RatingBias, RankBias


class Transformer(nn.Module):
    # def __init__(self, num_user, max_degree_user, num_item, max_degree_item, d_model, d_ffn, num_heads, dropout, num_layers_enc, num_layers_dec):
    def __init__(self, num_user, max_user_degree, num_item, max_item_degree, d_model, d_ffn, num_heads, dropout, enc_blocks, dec_blocks, n_experts, topk):
        super(Transformer, self).__init__()

        # embedding table 선언
        self.user_embed = SocialNodeEncoder(
            num_nodes = num_user,
            max_degree = max_user_degree,
            d_model = d_model)
        
        self.item_embed = ItemNodeEncoder(
            num_nodes = num_item,
            max_degree = max_item_degree,
            d_model = d_model
        )
        # encoder 선언
        self.encoder = Encoder(
            user_embed=self.user_embed,
            item_embed=self.item_embed,
            d_model=d_model,
            d_ffn=d_ffn,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=enc_blocks,
            n_experts=n_experts,
            topk=topk
        )
        # decoder 선언
        self.decoder = Decoder(
            user_embed=self.user_embed,
            item_embed=self.item_embed,
            d_model=d_model,
            d_ffn=d_ffn,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=dec_blocks,
            n_experts=n_experts,
            topk=topk
        )
    
    def forward(self, batched_data):
        enc_output, attention = self.encoder(batched_data)
        global_preference = attention
        
        output = self.decoder(batched_data, enc_output)
        output = torch.mean(output, dim=-1)
        # frobenius = torch.pow(self.user_embed.node_encoder.weight,2).sum() + torch.pow(self.item_embed.node_encoder.weight,2).sum()
        
        
        return output, global_preference