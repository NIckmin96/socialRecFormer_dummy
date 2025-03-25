import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.decoder_layer import DecoderLayer
from models.layers.encoding_modules import SocialNodeEncoder, ItemNodeEncoder, RatingEncoder, RatingBias, RankBias

from model_utils import generate_attn_pad_mask

class Decoder(nn.Module):
    """
    Decoder for modeling item representation (in user-item graph),
    and perform rating prediction
    """
    def __init__(self, num_user, num_item, max_user_degree, max_item_degree, d_model, d_ffn, num_heads, dropout, num_layers, n_experts, topk, rating_thres, args):
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
        
        # item embedding vector 생성
        self.item_embed = ItemNodeEncoder(
            num_nodes = num_item,
            max_degree = max_item_degree,
            d_model = d_model
        )
        # anchor user embedding vector 생성
        self.anchor_embed = SocialNodeEncoder(
            num_nodes = num_user,
            max_degree = max_user_degree,
            d_model = d_model
        )
        
        len_item_seq = args.user_seq_len*args.item_per_user
        # Rating embedding vector 생성
        self.rating_embed = RatingEncoder(num_user, len_item_seq, d_model)
        # Ranking attention bias(cross-attn1에 사용)
        self.ranking_bias = RankBias(rating_thres=rating_thres)
        # Rating attention bias(cross-attn2에 사용)
        self.rating_bias = RatingBias(num_heads=num_heads)

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(
                d_model = d_model,
                d_ffn = d_ffn,
                num_heads = num_heads,
                n_experts = n_experts,
                topk = topk,
                dropout = dropout,
                last_layer = False,
                is_dec_layer = True
            ) for _ in range(num_layers)]
        )

        # rating prediction layer(=last layer)
        self.pred_layer = DecoderLayer(
            d_model = d_model,
            d_ffn = d_ffn,
            num_heads = num_heads,
            dropout = dropout,
            last_layer = True,
            is_dec_layer = True
        )

        self.relu = nn.ReLU()
    
    def forward(self, batched_data, enc_output, user_embed, is_train):
        # Input Encoding: Node it encoding + degree encoding
            # [batch_size, seq_length, item_length]
        x_item = self.item_embed(batched_data['item_list'], batched_data['item_degree']) # bs x seq_len_item x d_model
        x_anchor = user_embed(batched_data['anchor_user'], batched_data['anchor_degree']).unsqueeze(1)  # bs x 1 x d_model
        x_anchor_i = self.item_embed(batched_data['anchor_items'], batched_data['anchor_item_degree']) # bs x seq_len_item x d_model
        rating_x = self.rating_embed(batched_data, is_train)
        device = rating_x.device

        # Generate mask for padded data
        self_attn_mask = generate_attn_pad_mask(batched_data['item_list'], batched_data['item_list']).to(device)    # [batch_size, seq_len_item, seq_len_item]
        cross_attn_mask_1 = generate_attn_pad_mask(batched_data['item_list'], batched_data['user_seq']).to(device) # [batch_size, seq_len_item, seq_len_user]
        cross_attn_mask_2 = generate_attn_pad_mask(batched_data['anchor_items'], batched_data['anchor_user'].unsqueeze(1)).to(device) # [bs x i x 1]

        ranking_bias = self.ranking_bias(batched_data['imp_fdback'])
        rating_bias = self.rating_bias(batched_data['item_rating'])
            
        rmse_losses = []
        # Decoder layer forward pass (MHA, FFN)
        for layer in self.dec_layers:
            x_item, rmse_loss, _ = layer(x_item, x_anchor, x_anchor_i, rating_x, enc_output, self_attn_mask, cross_attn_mask_1, cross_attn_mask_2, rating_bias, ranking_bias)
            rmse_losses.append(rmse_loss)
        
        # Pass to prediction layer
        output, rmse_loss, rating_pred = self.pred_layer(x_item, x_anchor, x_anchor_i, rating_x, enc_output, self_attn_mask, cross_attn_mask_1, cross_attn_mask_2, rating_bias, ranking_bias)
        rmse_losses.append(rmse_loss)
        
        # [bs, i, d] => [bs, i]
        output = torch.mean(output, dim=-1)
        # rank_logits = F.sigmoid(output)
        rank_logits = F.softmax(output, dim=-1)

        del self_attn_mask, cross_attn_mask_1, cross_attn_mask_2

        return rank_logits, rating_pred, sum(rmse_losses)/len(rmse_losses)