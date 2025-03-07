"""
Encoding & Embedding modules
"""
import numpy as np
import torch
import torch.nn as nn

class SocialNodeEncoder(nn.Module):
    def __init__(self, num_nodes, max_degree, d_model):
        super(SocialNodeEncoder, self).__init__()
        self.node_encoder = nn.Embedding(num_nodes + 1, d_model//2)
        self.degree_encoder = nn.Embedding(max_degree + 1, d_model//2, padding_idx=0)
    
    def forward(self, user_seq, user_degree):
        # Generate user_id embedding vector
        user_embedding = self.node_encoder(user_seq)
        degree_embedding = self.degree_encoder(user_degree)

        input_embedding = torch.cat([user_embedding, degree_embedding], dim=-1)

        return input_embedding

class SpatialEncoder(nn.Module):
    def __init__(self, num_heads, max_spd_value):
        super(SpatialEncoder, self).__init__()
        self.num_heads = num_heads

    def forward(self, batched_data):
        spd_matrix = batched_data['spd_matrix']
        bs,l,_ = spd_matrix.size()
        spd_matrix = spd_matrix.view(bs,1,l,l)
        attn_bias = spd_matrix.expand(-1, self.num_heads, -1, -1)

        return attn_bias

class ItemNodeEncoder(nn.Module):
    def __init__(self, num_nodes, max_degree, d_model):
        super(ItemNodeEncoder, self).__init__()
        self.node_encoder = nn.Embedding(num_nodes + 1, d_model//2)
        self.degree_encoder = nn.Embedding(max_degree + 1, d_model//2, padding_idx=0)
    
    def forward(self, item_seq, item_degree):
        item_embedding = self.node_encoder(item_seq)
        degree_embedding = self.degree_encoder(item_degree)

        input_embedding = torch.cat([item_embedding, degree_embedding], dim=-1)

        return input_embedding
    
class RatingEncoder(nn.Module):
    def __init__(self, num_nodes, len_item_seq, d_model):
        super(RatingEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.len_item_seq = len_item_seq
        self.user_bias = nn.Embedding(num_nodes+1, d_model) # 0 : cold start user
        self.rating_fc = nn.Linear(len_item_seq, d_model)

    def forward(self, batched_data, is_train=True):
        user_id = batched_data["user_seq"] # bs x u
        user_bias = self.user_bias(user_id)
        
        if is_train:
            item_rating = batched_data['item_rating'] # bs x u x i
            device = user_id.device

            bs,u,i = item_rating.size()
            if i != self.len_item_seq:
                index = torch.stack([torch.arange(i) for _ in range(u)]) # u x i
                index = torch.stack([index for _ in range(bs)]).to(device) # bs x u x i
                item_rating = torch.zeros(bs, u, self.num_items, dtype=item_rating.dtype, device=device).scatter(-1,index,item_rating) # num item 사이즈 맞추고 부족한 부분 zero padding
                item_rating = item_rating.float()

            
            rating_bias = self.rating_fc(item_rating.float())
            rating_embedding = (user_bias + rating_bias)
        else:
            rating_embedding = user_bias

        return rating_embedding


class RatingBias(nn.Module):
    def __init__(self, num_heads):
        super(RatingBias, self).__init__()
        self.num_heads = num_heads

    def forward(self, item_rating):
        item_rating = item_rating.permute(0,2,1) # (bs, i, u)
        item_rating = item_rating.unsqueeze(1) # (bs, 1, i, u)
        attn_bias = item_rating.expand(-1, self.num_heads, -1, -1) # (bs, h, i, u)

        return attn_bias
    
class RankBias(nn.Module):
    def __init__(self, rating_thres):
        super(RankBias, self).__init__()
        self.rating_thres=rating_thres

    def forward(self, imp_fdback):
        rank_bias = imp_fdback
        rank_bias = torch.where(rank_bias<self.rating_thres, 0, 1)

        return rank_bias