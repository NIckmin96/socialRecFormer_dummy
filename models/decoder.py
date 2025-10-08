import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.decoder_layer import DecoderLayer
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

from model_utils import generate_attn_pad_mask

class Decoder(nn.Module):
    """
    Decoder for modeling item representation (in user-item graph),
    and perform rating prediction
    """
    def __init__(self, user_seq_len, item_seq_len, user_embed, item_embed, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk, moe):
        """
        Args:
            data_path: path to dataset (ciao or epinions)
            num_item: number of total items in user-item graph (can be fetched from `degree_table_item.csv`)
            max_degree: max degree in user-item graph (also can be fetched from `degree_table_item.csv`)
            d_model: embedding dimension (attention module)
            d_ffn: embedding dimension (FFN module)
            num_heads: number of heads in multi-headed attention
            dropout: dropout rate
            num_layers: number of encoder layers
        """
        super(Decoder, self).__init__()
        self.user_embed = user_embed
        self.item_embed = item_embed
        
        if moe:
            moe_ffn = SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
        else:
            moe_ffn = FeedForwardNetwork(d_model, d_ffn, dropout)

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(
                user_seq_len = user_seq_len,
                item_seq_len = item_seq_len,
                d_model = d_model,
                d_ffn = d_ffn,
                num_heads = num_heads,
                n_experts = n_experts,
                topk = topk,
                dropout = dropout,
                # moe = moe
                moe = moe_ffn
            ) for _ in range(num_layers)]
        )
        
        # self.activation = nn.Sigmoid()
    
    def forward(self, batched_data, enc_output):
        # Input Encoding: Node it encoding + degree encoding
            # [batch_size, seq_length, item_length]
        device = batched_data['item_list'].device
        x_item = self.item_embed(batched_data['anchor_items'], batched_data['anchor_item_degree']) # bs x seq_len_item x d_model

        # Generate mask for padded data
        item_mask = generate_attn_pad_mask(batched_data['anchor_items'], batched_data['anchor_items']).to(device)    # [batch_size, seq_len_item, seq_len_item]
        item_user_mask = generate_attn_pad_mask(batched_data['anchor_items'], batched_data['anchor_user'].unsqueeze(1)).to(device) # [batch_size, seq_len_item, seq_len_user]
            
        # Decoder layer forward pass (MHA, FFN)
        for layer in self.dec_layers:
            x_item, attention, enc_output = layer(x_item, enc_output, item_mask, item_user_mask)
        
        # Mean
        # output = x_item
        # output = torch.mean(output, dim=-1)
        
        # MF
        enc_output = enc_output[:,0,:].unsqueeze(1)
        output = torch.matmul(enc_output, x_item.transpose(2,1)).squeeze(1)
        # normalize
        # output = self.activation(output)
        
        # attention의 첫번째 column = user representation과 item representation의 MM
        # output = attention[:,:,0]
        

        return output