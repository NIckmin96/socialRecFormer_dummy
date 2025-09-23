import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder

from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder, ItemNodeEncoder, RatingEncoder, RatingBias, RankBias


class Transformer(nn.Module):
    # def __init__(self, num_user, max_degree_user, num_item, max_degree_item, d_model, d_ffn, num_heads, dropout, num_layers_enc, num_layers_dec):
    def __init__(self, user_seq_len, item_seq_len, min_item_len, num_user, max_user_degree, num_item, max_item_degree, d_model, d_ffn, num_heads, dropout, enc_blocks, dec_blocks, n_experts, topk):
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
            user_seq_len=user_seq_len,
            item_seq_len=item_seq_len,
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
            user_seq_len=user_seq_len,
            item_seq_len=min_item_len, # anchor item 최대 길이
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
    
    def forward(self, batch):
        enc_output, global_preference = self.encoder(batch)
        output = self.decoder(batch, enc_output)
        
        
        return output, global_preference