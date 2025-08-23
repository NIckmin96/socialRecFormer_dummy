"""
trustnetwork 에서 모든 사용자의 degree 정보를 담은 table 생성

trustnetwork 에서 random walk sequence 생성
    - 임의의 사용자 n명을 선택
    - 각 사용자마다 random walk length r 만큼의 subgraph sequence 생성
    - 생성한 sequence에서, 각 노드와 매칭되는 degree 정보를 degree table에서 GET
    - [노드, 노드, 노드], [degree, degree, dgree] 를 함께 구성 (like PyG's edge_index)
        => [[node1, node2, node3]]
"""
import math
import os
# import time
import pickle
# import random
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import numpy as np
import pandas as pd
from ast import literal_eval    # convert str type list to original type
from scipy.io import loadmat
from tqdm.auto import tqdm
# from collections import defaultdict
from sklearn.utils import shuffle
import torch
from scipy import sparse
from data_preprocess import prepare_org_data

# 최초 한번만 실행
def mat_to_csv(data_path:str, regen=False):
    rating_path = os.path.join(data_path,'rating_new.csv')
    trust_path = os.path.join(data_path,'trustnetwork_new.csv')
    if os.path.isfile(rating_path) & os.path.isfile(trust_path) & (not regen):
        rating_df = pd.read_csv(rating_path)
        trust_df = pd.read_csv(trust_path)
    
    else:
        print("Creating rating_df, trust_df...")
        dataset_name = data_path.split('/')[-1]

        # original csv file
        rating_org = os.path.join(data_path, 'rating_org.csv')
        trust_org = os.path.join(data_path, 'trustnetwork_org.csv')
        if os.path.isfile(rating_org) & os.path.isfile(trust_org):
            rating_df = pd.read_csv(rating_org)
            trust_df = pd.read_csv(trust_org)
        else:
            rating_df, trust_df = prepare_org_data(data_path)

        rating_df = rating_df[['user_id','product_id','rating']]
        trust_df = trust_df[['user_id_1', 'user_id_2']]
        
        rating_df = rating_df.dropna(how='any')
        rating_df = rating_df.drop_duplicates(['user_id','product_id'], keep='first')
        rating_df = rating_df[rating_df.rating.between(1,5,inclusive='both')]
        trust_df = trust_df.dropna(how='any')
        trust_df = trust_df.drop_duplicates(keep='first')
        
        if dataset_name in ['yelp', 'Douban']:
            # 1. user당 rating이 너무 적은 경우 제외
            rating_df = rating_df.groupby('user_id').filter(lambda x: len(x) >= 10)
            # trust를 기준으로 user를 sampling
            trust_df = trust_df.sample(15000000, random_state=42, replace=False)
            
        rating_df, trust_df, rating_matrix = reset_and_filter_data(rating_df, trust_df)
        rating_df = add_degree(rating_df, trust_df)
        sparse.save_npz(os.path.join(data_path, 'rating_matrix.npz'), rating_matrix) # csr matrix 형태로 저장

        rating_df.to_csv(data_path + '/rating_new.csv', index=False)
        trust_df.to_csv(data_path + '/trustnetwork_new.csv', index=False)    

    return rating_df, trust_df

def reset_and_filter_data(rating_df:pd.DataFrame, trust_df:pd.DataFrame) -> pd.DataFrame:   
    total_users = set(trust_df.user_id_1.unique()).union(set(trust_df.user_id_2.unique())).intersection(set(rating_df.user_id.unique()))
    rating_df = rating_df[rating_df.user_id.isin(total_users)]
    trust_df = trust_df[trust_df.user_id_1.isin(total_users)|trust_df.user_id_2.isin(total_users)]
    
    # Generate user id mapping table
    total_users = set(trust_df.user_id_1.unique()).union(set(trust_df.user_id_2.unique())).union(set(rating_df.user_id.unique()))
    total_items = rating_df['product_id'].unique()
    user_dict = {user_id:idx+1 for idx,user_id in enumerate(total_users)}
    offset = max(user_dict.values())+1
    # Generate item id mapping table
    item_dict = {item_id:offset+idx for idx,item_id in enumerate(total_items)}
    print("user_dict: ", min(user_dict.values()), max(user_dict.values()))
    print("item_dict: ", min(item_dict.values()), max(item_dict.values()))

    rating_df['user_id']= rating_df['user_id'].map(user_dict)
    rating_df['product_id'] = rating_df['product_id'].map(item_dict)
    trust_df['user_id_1']= trust_df['user_id_1'].map(user_dict)
    trust_df['user_id_2'] = trust_df['user_id_2'].map(user_dict)
    
    # 전체 user-item rating 정보를 담은 rating matrix 생성
    rating_matrix = sparse.lil_matrix((rating_df['user_id'].nunique(), rating_df['product_id'].nunique()), dtype=np.ubyte)
    
    for _,row in rating_df.iterrows():
        u = row['user_id'].item()-1
        i = row['product_id'].item()-offset
        r = row['rating'].item()
        rating_matrix[u,i] = r
        
    rating_matrix = rating_matrix.tocsr()
    
    return rating_df, trust_df, rating_matrix

def add_degree(rating_df, trust_df):
    rating_g = nx.from_pandas_edgelist(rating_df, 'user_id', 'product_id')
    social_g = nx.from_pandas_edgelist(trust_df, 'user_id_1', 'user_id_2')
    
    rating_df['product_degree'] = rating_df['product_id'].map(lambda x:len(list(rating_g.neighbors(x))))
    rating_df['user_degree'] = rating_df['user_id'].map(lambda x:len(list(social_g.neighbors(x))))
    return rating_df

def shuffle_and_split_dataset(data_path:str, test=0.2, seed=42, regen=False):
    
    train_path = os.path.join(data_path, f'rating_train_seed_{seed}.csv')
    valid_path = os.path.join(data_path, f'rating_valid_seed_{seed}.csv')
    test_path = os.path.join(data_path, f'rating_test_seed_{seed}.csv')

    if os.path.isfile(train_path) & os.path.isfile(test_path) & (not regen):
        print("Loading Rating split sets...")
        rating_train_set = pd.read_csv(train_path)
        rating_valid_set = pd.read_csv(valid_path)
        rating_test_set = pd.read_csv(test_path)
        
    else:
        print("Creating Rating split sets...")
        rating_df = pd.read_csv(data_path + '/rating_new.csv', index_col=[])
        rating_df = rating_df.drop_duplicates(subset=['user_id','product_id'],keep='first')
        split_rating_df = shuffle(rating_df, random_state=seed)
        num_test = int(len(split_rating_df)*test)
        
        rating_test_set = split_rating_df.iloc[:num_test//2]
        rating_valid_set = split_rating_df.iloc[num_test//2:num_test]
        rating_train_set = split_rating_df.iloc[num_test:]

        rating_test_set.to_csv(data_path + f'/rating_test_seed_{seed}.csv', index=False)
        rating_valid_set.to_csv(data_path + f'/rating_valid_seed_{seed}.csv', index=False)
        rating_train_set.to_csv(data_path + f'/rating_train_seed_{seed}.csv', index=False)
    
    print(f"data split finished, seed: {seed}\n")
    
    return rating_train_set, rating_valid_set, rating_test_set

def generate_rw_sequence(data_path:str, rating_split:pd.DataFrame, walk_length:int, data_split_seed:int, split:str, regen:bool, used_set:set):
    
    user_counts = rating_split.user_id.value_counts()
    
    per_user = user_counts.quantile(0.25)
    print("per user : ", per_user)
    if data_path.split('/')[-1] in ['yelp']:
        per_user = min(user_counts.quantile(0.1),2)
        
    # anchor node list
    anchor_nodes = []
    for user,cnt in user_counts.items():
        k = int(min(per_user, cnt))
        anchor_nodes.extend([user]*k)
    
    file_path = os.path.join(data_path, f"new_rw_rating_length_{len(anchor_nodes)}_split_{split}_seed_{data_split_seed}.csv")
    # 이미 random walk 존재하는 경우 return
    if os.path.isfile(file_path)&(not regen):
        print(f"Loading {split} random walk sequence file...")
        df = pd.read_csv(file_path)
    # 새로 생성 or regen
    else:
        rating_graph = nx.from_pandas_edgelist(rating_split, 'user_id', 'product_id')
        user_degree_dic = rating_split.groupby('user_id')['user_degree'].unique().map(lambda x:int(x[0])).to_dict()
        item_degree_dic = rating_split.groupby('product_id')['product_degree'].unique().map(lambda x:int(x[0])).to_dict()
        degree_dic = dict()
        degree_dic.update(user_degree_dic)
        degree_dic.update(item_degree_dic)
        # random walk sequence
        
        anchor_seq_degree = []
        print(f"{split} random walk sequence file doesn't exist!")
        print(f"Creating {split} random walk sequence...")
        for node in tqdm(anchor_nodes, desc="Generating random walk sequence..."):
            seqs = [node]
            wl = 1
            cnt = 0
            cnt2 = 0
            while wl<walk_length:
                # 처음 : random next node 추출 후, append
                if len(seqs) <= 1:
                    next_node = find_next_node(rating_graph, previous_node=None, current_node=seqs[-1])
                # previous node(user면 user, item이면 item) 존재하는 경우
                else:
                    next_node = find_next_node(rating_graph, previous_node=seqs[-2], current_node=seqs[-1])
                    # 중복 노드가 존재하는 경우, 제외하고 random choice
                if next_node in seqs:
                    cnt2+=1
                    if cnt2>=5:
                        neighbors = list(rating_graph.neighbors(seqs[-1]))
                        next_node = np.random.choice(neighbors)
                        cnt2=0
                    else:
                        continue
                
                # user-item pair 중복 확인
                s = tuple(seqs[-2:])
                if s in used_set:
                    cnt+=1
                    if cnt<5:
                        continue
                    
                used_set.add(s)
                        
                seqs.append(next_node)
                wl += 1
            
                # revised
                degrees = []
                for node in seqs:
                    degrees.append(degree_dic.get(node,0))
                
            anchor_seq_degree.append([node,seqs,degrees])
            
        # revised
        df = pd.DataFrame(anchor_seq_degree,columns=['user_id','random_walk_seq','degree'])
        df = df.sort_values(by=['user_id'])
        df = df.reset_index(drop=True)
        df.to_csv(file_path, index=False)

    return df, file_path, used_set

def find_next_node(input_G, previous_node, current_node): 
    # neighbor구하기(user-> item / item->user)
        
    neighbors = list(input_G.neighbors(current_node))
    n = len(neighbors)
    # previous node가 지정된 경우 -> previous node 포함해서 확률 계산후, random choice
    if previous_node!=None:
        return_prob = 1/(n+max(n,2))
        edge_prob = (1-return_prob)/n
        # 정규화(sum=1)
        _sum = return_prob+n*edge_prob
        return_prob /= _sum
        edge_prob /= _sum

        candidates = neighbors + [previous_node]
        # print("candidates:", candidates)
        probs = [edge_prob for _ in range(n)]+[return_prob]
    # 지정된 previous node없는 경우, neighbor 중에서 random choice
    else:
        edge_prob = 1/n
        candidates = neighbors
        # print("candidates:", candidates)
        probs = [edge_prob for _ in range(n)]

    selected_node = np.random.choice(candidates, p=probs).item()
    
    return selected_node

def remove_duplicated_random_walk_sequence(random_walk_train:pd.DataFrame, random_walk_valid:pd.DataFrame, random_walk_test:pd.DataFrame, train_path:str, valid_path:str, test_path:str, regen:bool):
    if regen:
        random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]
        random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_valid['random_walk_seq'])]
        random_walk_valid = random_walk_valid[~random_walk_valid['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]

        random_walk_train.reset_index(drop=True, inplace=True); random_walk_valid.reset_index(drop=True, inplace=True); random_walk_test.reset_index(drop=True, inplace=True)

        random_walk_train.to_csv(train_path, index=False)
        random_walk_valid.to_csv(valid_path, index=False)
        random_walk_test.to_csv(test_path, index=False)

        print(f"rw_train len : {len(random_walk_train)} / rw_valid len : {len(random_walk_valid)} / rw_test len : {len(random_walk_test)}")

    return random_walk_train, random_walk_valid, random_walk_test
    
def generate_input_sequence_data(data_path, rw_df:pd.DataFrame, rating_df:pd.DataFrame, seed:int, split:str, random_walk_len:int, regen:bool):
    # test set augmentation 여부 확인
    total_path = data_path + f"/new_sequence_data_seed_{seed}_walk_{random_walk_len}_{split}.pkl"

    # total_df 재생성 여부 확인
    if os.path.isfile(total_path)&(not regen):
        print(f"Loading {split} total df(input_sequence_data)...")
        total_df = pd.read_pickle(total_path)
        print(f"total df dir : {total_path}")
        return total_df
    
    else:
        def str_to_list(x):
            return literal_eval(x) if type(x)==str else x
        
        tqdm.pandas()
        print(f"{split} total df(input_sequence_data) doesn't exist!")
        print(f"Creating {split} total df(input_sequence_data)...")
        # informations
        rating_matrix = sparse.load_npz(os.path.join(data_path, 'rating_matrix.npz')) # main으로
        user_degree_dic = rating_df.groupby('user_id')['user_degree'].unique().map(lambda x:int(x[0])).to_dict()
        product_degree_dic = rating_df.groupby('product_id')['product_degree'].unique().map(lambda x:int(x[0])).to_dict()
        interacted_items = rating_df.groupby('user_id')['product_id'].unique().to_dict()
        offset = rating_df.product_id.min()
        print(rating_df.user_id.min(), rating_df.user_id.max())
        print(rating_df.product_id.min(), rating_df.product_id.max())
        total_df = pd.DataFrame()
        # rw sequence
        total_df['rw_seq'] = rw_df['random_walk_seq'].progress_map(str_to_list)       
        total_df['rw_degree'] = rw_df['degree'].progress_map(str_to_list)
        # interacted items
        total_df['user'] = total_df['rw_seq'].progress_map(lambda x:x[0])
        total_df['product'] = total_df['user'].progress_map(lambda x:interacted_items[x])
        total_df['user_offset'] = total_df['user'].progress_map(lambda x:x-1)
        total_df['product_offset'] = total_df['product'].progress_map(lambda seq:list(map(lambda x:x-offset, seq)))

        total_df['user_degree'] = total_df['user'].progress_map(lambda x:user_degree_dic[x])
        total_df['product_degree'] = total_df['product'].progress_map(lambda seq:list(map(lambda x:product_degree_dic[x], seq)))
        total_df['ratings'] = total_df.progress_apply(lambda x:torch.LongTensor(rating_matrix[x['user_offset'],x['product_offset']].toarray().astype(int)), axis=1)
    
        total_df = total_df[['rw_seq','rw_degree','user','product','user_degree','product_degree','ratings']]

        with open(total_path, "wb") as file:
            pickle.dump(total_df, file)

        print(f"# of total {split} : {len(total_df)}")    
        
        return total_df
    