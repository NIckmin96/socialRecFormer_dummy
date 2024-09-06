import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
    
    def __init__(self, dataframe):

        self.user_sequences = torch.tensor(dataframe['user_sequences'].tolist(), dtype=torch.long)

        self.user_degree = torch.tensor(dataframe['user_degree'].tolist(), dtype=torch.long)

        self.item_sequences = torch.tensor(dataframe['item_sequences'].tolist(), dtype=torch.long)

        self.item_degree = torch.tensor(dataframe['item_degree'].tolist(), dtype=torch.long)

        self.rating_matrix = dataframe['item_rating']
        
        self.spd_matrix = dataframe['spd_matrix']
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
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
        
# import os
# import pickle
# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class MyDataset(Dataset):
#     """
#     Process & make dataset for DataLoader
#     """
#     def __init__(self, dataset:str, split:str, seed:int, user_seq_len:int=20, item_seq_len:int=250, return_params:int=1):
#         """
#         Args:
#             dataset: raw dataset name (ciao // epinions)
#             split: dataset split type (train // valid // test)
#             seed: random seed used in dataset split
#             item_seq_len: length of item list (processed in `data_utils.py`)
#         """
#         self.data_path = os.path.join(os.getcwd(), 'dataset', dataset)
        
#         # Load preprocessed .pkl file
#         file_path = os.path.join(
#             self.data_path, 
#             f'sequence_data_seed_{seed}_walk_{user_seq_len}_itemlen_{item_seq_len}_rp_{return_params}_{split}.pkl'
#         )
#         dataframe = pd.read_pickle(file_path)
#         print("dataset loaded")

#         self.user_sequences = torch.tensor(dataframe['user_sequences'].tolist(), dtype=torch.long)

#         self.user_degree = torch.tensor(dataframe['user_degree'].tolist(), dtype=torch.long)

#         self.item_sequences = torch.tensor(dataframe['item_sequences'].tolist(), dtype=torch.long)

#         self.item_degree = torch.tensor(dataframe['item_degree'].tolist(), dtype=torch.long)

#         print("start")

#         self.rating_matrix = dataframe['item_rating']
        
#         self.spd_matrix = dataframe['spd_matrix']

#         print("end")

    
#     def __len__(self):
#         return len(self.user_sequences)

#     def __getitem__(self, index):
#         return {
#             'user_seq': self.user_sequences[index],
#             'user_degree': self.user_degree[index],
#             'item_list': self.item_sequences[index],
#             'item_degree': self.item_degree[index],
#             'item_rating': self.rating_matrix[index],
#             'spd_matrix': self.spd_matrix[index]
#         }
    
# if __name__ == "__main__":
#     dataset = 'ciao'
#     split = 'train'

#     dataset = MyDataset(dataset, split, seed=42, user_seq_len=30, item_seq_len=100)
#     print("?")
#     loader = DataLoader(dataset, batch_size=256, shuffle=True)
#     print("?")
#     for data in loader:
#         # print(data['user_seq'].shape)
#         # print(data['user_degree'].shape)
#         # print(data['item_list'].shape)
#         # print(data['item_degree'].shape)
#         # print(data['item_rating'].shape)
#         # print(data['spd_matrix'].shape)
#         data['item_rating'] = data['item_rating'].view(-1)          # convert to 2D tensor
#         print(torch.bincount(data['item_rating']))
#         # print(data['user_seq'][0])
#         # print(data['user_degree'][0])
#         # print(data['item_list'][0])
#         # print(data['item_degree'][0])
#         # print(data['item_rating'][0])
#         # print(data['spd_matrix'][0])
#         quit()