import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, dataframe):
        # [DEV]
        self.rw_sequence = torch.tensor(dataframe['rw_sequence'], dtype=torch.long)
        self.degree = torch.tensor(dataframe['degree'], dtype=torch.long)
        
        self.anchor_user = torch.tensor(dataframe['user_id'], dtype=torch.long)
        self.anchor_degree = torch.tensor(dataframe['anchor_degree'], dtype=torch.long)
        self.anchor_items = torch.tensor(dataframe['anchor_items'].tolist(), dtype=torch.long)        
        self.anchor_item_degree = torch.tensor(dataframe['anchor_item_degree'].tolist(), dtype=torch.long)
        self.imp_fdback = torch.tensor(dataframe['imp_fdback'].tolist(), dtype=torch.long)
        self.exp_fdback = torch.tensor(dataframe['anchor_ratings'].tolist(), dtype=torch.long)
        
        # [ORG]
        self.user_sequences = torch.tensor(dataframe['user_sequences'].tolist(), dtype=torch.long)
        self.user_degree = torch.tensor(dataframe['user_degree'].tolist(), dtype=torch.long)
        self.item_sequences = torch.tensor(dataframe['item_sequences'].tolist(), dtype=torch.long)
        self.item_degree = torch.tensor(dataframe['item_degree'].tolist(), dtype=torch.long)
        self.rating_matrix = dataframe['item_rating']
        # self.spd_matrix = dataframe['spd_matrix']
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
        return len(self.user_sequences)

    def __getitem__(self, index):
        return {
            # [DEV]
            'anchor_user' : self.anchor_user[index],
            'anchor_degree' : self.anchor_degree[index],
            'anchor_items':self.anchor_items[index],
            'anchor_item_degree':self.anchor_item_degree[index],
            'imp_fdback':self.imp_fdback[index],
            'exp_fdback':self.exp_fdback[index],
            # [ORG]
            'user_seq': self.user_sequences[index],
            'user_degree': self.user_degree[index],
            'item_list': self.item_sequences[index],
            'item_degree': self.item_degree[index],
            'item_rating': self.rating_matrix[index],
            # 'spd_matrix': self.spd_matrix[index]
        }