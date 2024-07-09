import os
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# class MyDataset(Dataset):
#     """
#     Process & make dataset for DataLoader
#         __init__():     data load & convert to Tensor
#         __getitem__():  data indexing (e.g. dataset[0])
#         __len__():      length of dataset (e.g. len(dataset))
#     """
#     def __init__(self, dataset:str, split:str, seed:int, user_seq_len:int=20, item_seq_len:int=250, return_params:int=1):
#         """
#         Args:
#             dataset: raw dataset name (ciao // epinions)
#             split: dataset split type (train // valid // test)
#             seed: random seed used in dataset split
#             item_seq_len: length of item list (processed in `data_utils.py`)
#         """
#         self.data_path = os.getcwd() + '/dataset/' + dataset

#         # 전처리 된 .pkl 파일 load
#         # with open(self.data_path + '/' f'sequence_data_seed_{seed}_walk_{user_seq_len}_itemlen_{item_seq_len}_rp_{return_params}_{split}.pkl', 'rb') as file:
#         #     dataframe = pickle.load(file)
#         dataframe = pd.read_pickle(self.data_path + '/' f'sequence_data_seed_{seed}_walk_{user_seq_len}_itemlen_{item_seq_len}_rp_{return_params}_{split}.pkl')
#         print("loaded")

#         user_sequences = dataframe['user_sequences'].values
#         user_sequences = np.array([np.array(x) for x in user_sequences])

#         user_degree = dataframe['user_degree'].values
#         user_degree = np.array([np.array(x) for x in user_degree])

#         item_sequences = dataframe['item_sequences'].values
#         item_sequences = np.array([np.array(x) for x in item_sequences])

#         item_degree = dataframe['item_degree'].values
#         item_degree = np.array([np.array(x) for x in item_degree])

#         # shape: [total_samples(num_row), seq_len_user, seq_len_item]
#         rating_matrix = dataframe['item_rating'].values
#         rating_matrix = np.array([np.array(x) for x in rating_matrix])

#         # shape: [total_samples(num_row), seq_len_user, seq_len_user]
#         spd_matrix = dataframe['spd_matrix'].values
#         spd_matrix = np.array([np.array(x) for x in spd_matrix])

#         self.user_sequences = torch.LongTensor(user_sequences)
#         self.user_degree = torch.LongTensor(user_degree)
#         self.item_sequences = torch.LongTensor(item_sequences)
#         self.item_degree = torch.LongTensor(item_degree)
#         self.rating_matrix = torch.LongTensor(rating_matrix)
#         self.spd_matrix = torch.LongTensor(spd_matrix)
    
#     def __len__(self):
#         # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
#         return len(self.user_sequences)

#     def __getitem__(self, index):
#         user_seq = self.user_sequences[index]
#         user_deg = self.user_degree[index]
#         item_seq = self.item_sequences[index]
#         item_deg = self.item_degree[index]
#         rating_table = self.rating_matrix[index]
#         spd_table = self.spd_matrix[index]

#         batch_data = {
#             'user_seq': user_seq,
#             'user_degree': user_deg,
#             'item_list': item_seq,
#             'item_degree': item_deg,
#             'item_rating': rating_table,
#             'spd_matrix': spd_table
#         }

#         return batch_data
        
import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Process & make dataset for DataLoader
    """
    def __init__(self, dataset:str, split:str, seed:int, user_seq_len:int=20, item_seq_len:int=250, return_params:int=1):
        """
        Args:
            dataset: raw dataset name (ciao // epinions)
            split: dataset split type (train // valid // test)
            seed: random seed used in dataset split
            item_seq_len: length of item list (processed in `data_utils.py`)
        """
        self.data_path = os.path.join(os.getcwd(), 'dataset', dataset)
        
        # Load preprocessed .pkl file
        file_path = os.path.join(
            self.data_path, 
            f'sequence_data_seed_{seed}_walk_{user_seq_len}_itemlen_{item_seq_len}_rp_{return_params}_{split}.pkl'
        )
        dataframe = pd.read_pickle(file_path)
        print("dataset loaded")

        self.user_sequences = torch.tensor(dataframe['user_sequences'].tolist(), dtype=torch.long)

        self.user_degree = torch.tensor(dataframe['user_degree'].tolist(), dtype=torch.long)

        self.item_sequences = torch.tensor(dataframe['item_sequences'].tolist(), dtype=torch.long)

        self.item_degree = torch.tensor(dataframe['item_degree'].tolist(), dtype=torch.long)

        print("start")

        self.rating_matrix = dataframe['item_rating']
        
        self.spd_matrix = dataframe['spd_matrix']

        print("end")

    
    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, index):
        return {
            'user_seq': self.user_sequences[index],
            'user_degree': self.user_degree[index],
            'item_list': self.item_sequences[index],
            'item_degree': self.item_degree[index],
            'item_rating': self.rating_matrix[index],
            'spd_matrix': self.spd_matrix[index]
        }
    
if __name__ == "__main__":
    dataset = 'ciao'
    split = 'train'

    dataset = MyDataset(dataset, split, seed=42, user_seq_len=30, item_seq_len=100)
    print("?")
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print("?")
    for data in loader:
        # print(data['user_seq'].shape)
        # print(data['user_degree'].shape)
        # print(data['item_list'].shape)
        # print(data['item_degree'].shape)
        # print(data['item_rating'].shape)
        # print(data['spd_matrix'].shape)
        data['item_rating'] = data['item_rating'].view(-1)          # convert to 2D tensor
        print(torch.bincount(data['item_rating']))
        # print(data['user_seq'][0])
        # print(data['user_degree'][0])
        # print(data['item_list'][0])
        # print(data['item_degree'][0])
        # print(data['item_rating'][0])
        # print(data['spd_matrix'][0])
        quit()