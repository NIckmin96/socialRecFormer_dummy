import torch
from torch.utils.data import Dataset

class RWDataset(Dataset):
    
    def __init__(self, total_df):
        self.rw_seq = total_df['rw_seq'].values
        self.rw_degree = total_df['rw_degree'].values
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
        return len(self.rw_seq)

    def __getitem__(self, index):
        return {
            'rw_seq':torch.tensor(self.rw_seq[index]).long(),
            'rw_degree':torch.tensor(self.rw_degree[index]).long()
        }
        
class AnchorDataset(Dataset):
    def __init__(self, total_df):
        self.user = total_df['user'].values
        self.user_degree = total_df['user_degree'].values
        self.product = total_df['product'].values
        self.product_degree = total_df['product_degree'].values
        self.ratings = total_df['ratings'].values
    
    def __len__(self):
        # 전체 {train/valid/test}.csv의 길이 (dataframe의 전체 row 갯수)
        return len(self.product)

    def __getitem__(self, index):
        data =  {
            'user':torch.tensor(self.user[index]).long(),
            'user_degree':torch.tensor(self.user_degree[index]).long(),
            'product':torch.tensor(self.product[index]).long(),
            'product_degree':torch.tensor(self.product_degree[index]).long(),
            'ratings':torch.tensor(self.ratings[index]).float()
        }
        # print(data['user'].shape, data['product'].shape, data['product_degree'].shape, data['ratings'].shape)
        
        return data