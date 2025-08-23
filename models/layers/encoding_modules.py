"""
Encoding & Embedding modules
"""
import numpy as np
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, max_node, max_degree, d_model):
        super(Embedding, self).__init__()
        self.node_encoder = nn.Embedding(max_node + 1, d_model//2)
        self.degree_encoder = nn.Embedding(max_degree + 1, d_model//2, padding_idx=0)
    
    def forward(self, node, degree):
        node_embedding = self.node_encoder(node)
        degree_embedding = self.degree_encoder(degree)
        input_embedding = torch.cat([node_embedding, degree_embedding], dim=-1)

        return input_embedding