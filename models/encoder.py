import torch.nn as nn

from models.blocks.encoder_layer import EncoderLayer
from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder

from model_utils import generate_attn_pad_mask

class Encoder(nn.Module):
    """
    Encoder for modeling user representation (in social graph)
    """
    # def __init__(self, max_degree, num_user, d_model, d_ffn, num_heads, dropout, num_layers):
    def __init__(self, max_degree, num_user, max_spd_value, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk):
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

        self.max_degree = max_degree
        self.num_user = num_user
        self.max_spd_value = max_spd_value

        self.input_embed = SocialNodeEncoder(
            num_nodes = self.num_user,
            max_degree = self.max_degree,
            d_model = d_model
        )

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(
                d_model = d_model,
                d_ffn = d_ffn,
                num_heads = num_heads,
                n_experts = n_experts,
                topk = topk,
                dropout = dropout
            ) for _ in range(num_layers)]
        )

        # self.spatial_pos_bias = SpatialEncoder(
        #     # num_nodes = self.num_user,
        #     max_spd_value = self.max_spd_value,
        #     num_heads = num_heads
        # )
    
    def forward(self, batched_data):
        x = self.input_embed(batched_data['user_seq'], batched_data['user_degree'])

        # Generate mask for padded data
        src_mask = generate_attn_pad_mask(batched_data['user_seq'], batched_data['user_seq'])
        # attn_bias = self.spatial_pos_bias(batched_data)

        losses = []
        # Encoder layer forward pass (MHA, FFN)
        for layer in self.enc_layers:
            x, spd_loss = layer(x, src_mask, None)
            # x, spd_loss = layer(x, src_mask, None)
            losses.append(spd_loss)

        
        return x, sum(losses)/len(losses), self.input_embed