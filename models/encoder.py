import torch
import torch.nn as nn

from models.blocks.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Encoder for modeling user representation (in social graph)
    """
    def __init__(self, input_embed, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk):
        super(Encoder, self).__init__()

        self.input_embed = input_embed

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
    
    def forward(self, batched_data):
        x = self.input_embed(batched_data['rw_seq'], batched_data['rw_degree'])
        # Encoder layer forward pass (MHA, FFN)
        for layer in self.enc_layers:
            x = layer(x)
        
        return x