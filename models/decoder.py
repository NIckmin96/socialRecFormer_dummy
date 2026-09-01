import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.decoder_layer import DecoderLayer
from models.layers.feed_forward_network import FeedForwardNetwork, SparseMoE

from model_utils import generate_attn_pad_mask

class Decoder(nn.Module):
    def __init__(self, user_seq_len, item_seq_len, user_embed, item_embed, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk, moe):
        super(Decoder, self).__init__()
        self.user_embed = user_embed
        self.item_embed = item_embed

        # Pre-LN 구조에서 마지막 블록 출력의 residual stream을 안정화하기 위한 최종 정규화
        self.final_norm = nn.LayerNorm(d_model)

        def make_moe_ffn():
            # 블록마다 독립된 MoE/FFN 파라미터를 갖도록 매번 새로 생성
            if moe:
                return SparseMoE(d_model, d_ffn, n_experts, topk, dropout)
            else:
                return FeedForwardNetwork(d_model, d_ffn, dropout)

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(
                user_seq_len = user_seq_len,
                item_seq_len = item_seq_len,
                d_model = d_model,
                num_heads = num_heads,
                dropout = dropout,
                moe = make_moe_ffn()
            ) for _ in range(num_layers)]
        )
    
    def forward(self, batched_data, enc_output):
        device = batched_data['item_list'].device
        x_item = self.item_embed(batched_data['anchor_items'], batched_data['anchor_item_degree']) # bs x seq_len_item x d_model

        # Generate mask for padded data
        item_mask = generate_attn_pad_mask(batched_data['anchor_items'], batched_data['anchor_items']).to(device)    # [batch_size, seq_len_item, seq_len_item]
        item_user_mask = generate_attn_pad_mask(batched_data['anchor_items'], batched_data['anchor_user'].unsqueeze(1)).to(device) # [batch_size, seq_len_item, seq_len_user]
            
        # Decoder layer forward pass (MHA, FFN)
        for layer in self.dec_layers:
            x_item, attention, enc_output = layer(x_item, enc_output, item_mask, item_user_mask)

        self.user_reptn = enc_output

        # 마지막 블록 이후 최종 정규화 (item representation stream)
        x_item = self.final_norm(x_item)

        # MF
        enc_output = enc_output[:,0,:].unsqueeze(1)
        output = torch.matmul(enc_output, x_item.transpose(2,1)).squeeze(1)

        return output