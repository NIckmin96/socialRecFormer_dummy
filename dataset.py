import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, total_df, total_df2):
        # anchor
        self.anchor_user = torch.tensor(total_df2['anchor_user'].tolist(), dtype=torch.long)
        self.anchor_degree = torch.tensor(total_df2['anchor_degree'].tolist(), dtype=torch.long)
        self.anchor_items = torch.tensor(total_df2['anchor_items'].tolist(), dtype=torch.long)        
        self.anchor_item_degree = torch.tensor(total_df2['anchor_item_degree'].tolist(), dtype=torch.long)
        self.anchor_ratings = torch.stack(total_df2['anchor_ratings'].tolist()).squeeze(1)
        
        # sequence
        self.user_sequences = torch.tensor(total_df['user_sequences'].tolist(), dtype=torch.long)
        self.user_degree = torch.tensor(total_df['user_degree'].tolist(), dtype=torch.long)
        self.item_sequences = torch.tensor(total_df['item_sequences'].tolist(), dtype=torch.long)
        self.item_degree = torch.tensor(total_df['item_degree'].tolist(), dtype=torch.long)
        self.item_rating = torch.stack(total_df['item_rating'].tolist()).squeeze(1)
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
        return len(self.user_sequences)

    def __getitem__(self, index):
        return {
            # anchor
            'anchor_user' : self.anchor_user[index],
            'anchor_degree' : self.anchor_degree[index],
            'anchor_items':self.anchor_items[index],
            'anchor_item_degree':self.anchor_item_degree[index],
            'anchor_ratings':self.anchor_ratings[index],
            # sequence
            'user_seq': self.user_sequences[index],
            'user_degree': self.user_degree[index],
            'item_list': self.item_sequences[index],
            'item_degree': self.item_degree[index],
            'item_rating': self.item_rating[index],
            
        }