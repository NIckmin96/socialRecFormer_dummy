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
def mat_to_csv(data_path:str, regen):
    rating_path = os.path.join(data_path,'rating.csv')
    trust_path = os.path.join(data_path,'trustnetwork.csv')
    if os.path.isfile(rating_path) & os.path.isfile(trust_path) & (regen!='all'):
        rating_df = pd.read_csv(rating_path)
        trust_df = pd.read_csv(trust_path)
    
    else:
        print("Creating rating_df, trust_df...")
        dataset_name = data_path.split('/')[-1]

        # original csv file
        rating_org = os.path.join(data_path, 'rating_org.csv')
        trust_org = os.path.join(data_path, 'trustnetwork_org.csv')
        if os.path.isfile(rating_org) & os.path.isfile(trust_org) & (regen):
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
        
        if dataset_name in ['yelp']:
            # 1. user당 rating이 너무 적은 경우 제외
            rating_df = rating_df.groupby('user_id').filter(lambda x: len(x) >= 10)
            # trust를 기준으로 user를 sampling
            trust_df = trust_df.sample(15000000, random_state=42, replace=False)
            
        rating_df, trust_df = reset_and_filter_data(rating_df, trust_df)
        rating_df = add_degree(rating_df, trust_df)

        # 전체 user-item rating 정보를 담은 rating matrix 생성
        rating_matrix = sparse.lil_matrix((max(rating_df['user_id'].unique())+1, max(rating_df['product_id'].unique())+1), dtype=np.ubyte)

        for index in rating_df.index:
            rating_matrix[rating_df['user_id'][index], rating_df['product_id'][index]] = rating_df['rating'][index]
            
        rating_matrix = rating_matrix.tocsr()
        sparse.save_npz(os.path.join(data_path, 'rating_matrix.npz'), rating_matrix) # csr matrix 형태로 저장

        rating_df.to_csv(data_path + '/rating.csv', index=False)
        trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)
    
    # data statistics
    print(f"***** Dataset Statistics *****")
    print(f"# of users : {max(rating_df.user_id.max(), trust_df.user_id_1.max(), trust_df.user_id_2.max())}")
    print(f"# of users in rating df : {rating_df.user_id.nunique()}")
    assert rating_df.product_id.nunique()==rating_df.product_id.max()
    print(f"# of items : {rating_df.product_id.nunique()}")
    print(f"# of interactions : {rating_df.shape[0]}")
    print(f"# of Social Links : {trust_df.shape[0]}")

    return rating_df, trust_df

def reset_and_filter_data(rating_df:pd.DataFrame, trust_df:pd.DataFrame) -> pd.DataFrame:   
    # filter data by users(existing in both columns in trust_df)
    total_users = set(trust_df.user_id_1.unique()).union(set(trust_df.user_id_2.unique())).intersection(set(rating_df.user_id.unique()))
    rating_df = rating_df[rating_df.user_id.isin(total_users)]
    trust_df = trust_df[trust_df.user_id_1.isin(total_users)&trust_df.user_id_2.isin(total_users)]
    
    # Generate user id mapping table
    total_users = set(trust_df.user_id_1.unique()).union(set(trust_df.user_id_2.unique())).union(set(rating_df.user_id.unique()))
    mapping_table_user = {user_id:idx+1 for idx,user_id in enumerate(total_users)}
    # Generate item id mapping table
    mapping_table_item = {item_id:idx+1 for idx,item_id in enumerate(rating_df['product_id'].unique())}    

    rating_df['user_id']= rating_df['user_id'].map(mapping_table_user)
    rating_df['product_id'] = rating_df['product_id'].map(mapping_table_item)
    trust_df['user_id_1']= trust_df['user_id_1'].map(mapping_table_user)
    trust_df['user_id_2'] = trust_df['user_id_2'].map(mapping_table_user)

    return rating_df, trust_df

def add_degree(rating_df, trust_df):
    rating_g = nx.from_pandas_edgelist(rating_df, 'user_id', 'product_id')
    social_g = nx.from_pandas_edgelist(trust_df, 'user_id_1', 'user_id_2')
    social_degree = {user:social_g.degree(user) for user in social_g.nodes()}
    
    rating_df['product_degree'] = rating_df['product_id'].map(lambda x:len(list(rating_g.neighbors(x))))
    rating_df['user_degree'] = rating_df['user_id'].map(lambda x:social_degree.get(x,0))
    return rating_df


def shuffle_and_split_dataset(data_path:str, test, seed, regen):
    
    train_path = os.path.join(data_path, f'rating_train_seed_{seed}.csv')
    valid_path = os.path.join(data_path, f'rating_valid_seed_{seed}.csv')
    test_path = os.path.join(data_path, f'rating_test_seed_{seed}.csv')

    # if (os.path.isfile(train_path)&os.path.isfile(valid_path)&os.path.isfile(test_path)&(not regen)):
    if os.path.isfile(train_path) & os.path.isfile(test_path) & (regen != 'all'):
        print("Loading Rating split sets...")
        rating_train_set = pd.read_csv(train_path)
        rating_valid_set = pd.read_csv(valid_path)
        rating_test_set = pd.read_csv(test_path)
        
    else:
        print("Creating Rating split sets...")
        rating_df = pd.read_csv(data_path + '/rating.csv', index_col=[])
        rating_df = rating_df.drop_duplicates(subset=['user_id','product_id'],keep='first')
        split_rating_df = shuffle(rating_df, random_state=seed)
        num_test = int(len(split_rating_df)*test)
        
        # rating_test_set = split_rating_df.iloc[:num_test]
        # rating_train_set = split_rating_df.iloc[num_test:]
        
        rating_test_set = split_rating_df.iloc[:num_test//2]
        rating_valid_set = split_rating_df.iloc[num_test//2:num_test]
        rating_train_set = split_rating_df.iloc[num_test:]

        rating_test_set.to_csv(data_path + f'/rating_test_seed_{seed}.csv', index=False)
        rating_valid_set.to_csv(data_path + f'/rating_valid_seed_{seed}.csv', index=False)
        rating_train_set.to_csv(data_path + f'/rating_train_seed_{seed}.csv', index=False)
    
    print(f"data split finished, seed: {seed}\n")
    
    return rating_train_set, rating_valid_set, rating_test_set
    # return rating_train_set, rating_test_set

def generate_social_dataset(data_path, split, rating_split, trust_df, seed, regen):
    """
    Generate social graph from train/test/validation dataset
    """
    social_file = os.path.join(data_path, f'trustnetwork_{split}_seed_{seed}.csv')
    rating_file = os.path.join(data_path, f'rating_{split}_seed_{seed}.csv')
    if (not os.path.isfile(social_file)) or (regen=='all'):
        print(f"Creating Social {split} split sets...\n")
        users = rating_split['user_id'].unique()            
        social_split = trust_df[(trust_df['user_id_1'].isin(users)) & (trust_df['user_id_2'].isin(users))]

        # save
        social_split.to_csv(social_file, index=False)
        rating_split.to_csv(rating_file, index=False)
    else:
        print("Loading Social split sets...\n")
        social_split = pd.read_csv(social_file)
        rating_split = pd.read_csv(rating_file)
    
    return social_split, rating_split

def generate_social_random_walk_sequence(data_path, rating_split, social_split, walk_length, data_split_seed, split, augs, regen):
    # rating split -> social split : rating에 존재하는 user를 기준으로 social split 생성 
    # social split -> social graph : social split을 기준으로 graph 생성 -> graph의 전체 node를 순회하면서 random walk 생성 -> rating안에 존재하는 user가 아닌 경우에 item이 붙을 수가 없음 -> 불필요한 데이터 생성 -> rating 기준이 맞음!!
    # experiment : social node 전체 순회 vs rating split user node기준 순회
    social_graph = nx.from_pandas_edgelist(social_split, source='user_id_1', target='user_id_2')
    anchor_nodes = list(social_graph.nodes())*augs
        
    # save dir 지정
    if split=='train':
        file_path = os.path.join(data_path, f"new_rw_rating_length_{len(anchor_nodes)}_split_{split}_seed_{data_split_seed}_{augs}.csv")
    else:
        file_path = os.path.join(data_path, f"new_rw_rating_length_{len(anchor_nodes)}_split_{split}_seed_{data_split_seed}.csv")
    # 이미 random walk 존재하는 경우 return
    if os.path.isfile(file_path) & (regen in ['no','total']):
        print(f"Loading {split} random walk sequence file...")
        df = pd.read_csv(file_path)
    # 새로 생성 or regen
    else:
        user_degree_dic = rating_split.groupby('user_id')['user_degree'].unique().map(lambda x:int(x[0])).to_dict()
        # random walk sequence
        # anchor_cnt = {n:0 for n in rating_users}
        
        anchor_seq_degree = []
        # seq_set = set()
        print(f"{split} random walk sequence file doesn't exist!")
        print(f"Creating {split} random walk sequence...")
        for node in tqdm(anchor_nodes, desc="Generating random walk sequence..."):
            # if anchor_cnt[node]==per_user:
            #     continue
            seqs = [node]
            wl = 1
            thres=0
            while wl<walk_length:
                # 처음 : random next node 추출 후, append
                if wl == 1:
                    next_node = find_next_node(social_graph, previous_node=None, current_node=node)
                # 처음이 아닌 경우
                else:
                    next_node = find_next_node(social_graph, previous_node=seqs[-2], current_node=seqs[-1])
                    # 중복 노드가 존재하는 경우, 제외하고 random choice
                    if next_node in seqs:
                        if thres<5:
                            thres+=1
                            continue
                        else:
                            available = set(social_graph.nodes())-set(seqs)
                            if available:
                                next_node = np.random.choice(list(available))
                            else:
                                next_node = np.random.choice(list(social_graph.nodes()))

                # next node 추가 후, walk length 하나 올림
                seqs.append(next_node)
                wl += 1
            
            # anchor_cnt[node]+=1
            degrees = [0 if node==0 else user_degree_dic.get(node, 0) for node in seqs]
            anchor_seq_degree.append([node,seqs,degrees])

        # revised
        df = pd.DataFrame(anchor_seq_degree,columns=['user_id','random_walk_seq','degree'])
        df = df.sort_values(by=['user_id'])
        df = df.reset_index(drop=True)
        df.to_csv(file_path, index=False)

    return df, file_path

def find_next_node(input_G, previous_node, current_node): # 확률적으로, anchor node가 동일하다면 중복되는 random walk sequence가 나올수도 있음
    # 문제 : neighbor가 많을 경우에, 이전 노드로 돌아갈 확률이 다른 노드로 갈 확률보다 높아짐 -> 의도된 것?
    # return param을 고정하지않고, neighbor의 개수에 따라 유동적으로 변하는게 합리적임 -> n개의 neighbor가 있으면, x = (1/n)*n + return, 1 = (1/nx)*n + return/x
    if previous_node!=None:
        if current_node in input_G.nodes():
            neighbors = list(set(input_G.neighbors(current_node))-{previous_node})
        else:
            neighbors = list(set(input_G.nodes())-{current_node})
        
        n = len(neighbors)
        if n==0:
            return np.random.choice(input_G.nodes()).item()
        
        return_prob = 1/(n+max(n,2))
        edge_prob = (1-return_prob)/n
        # 정규화(sum=1)
        _sum = return_prob+n*edge_prob
        return_prob /= _sum
        edge_prob /= _sum

        candidates = neighbors + [previous_node]
        probs = [edge_prob for _ in range(n)]+[return_prob]
        
    else:
        if current_node in input_G.nodes():
            neighbors = list(set(input_G.neighbors(current_node))-{previous_node})
        else:
            neighbors = list(set(input_G.nodes())-{current_node})
        n = len(neighbors)
        
        edge_prob = 1/n
        candidates = neighbors
        probs = [edge_prob for _ in range(n)]

    selected_node = np.random.choice(candidates, p=probs).item()
    
    return selected_node

def remove_duplicated_social_random_walk_sequence(random_walk_train:pd.DataFrame, random_walk_valid:pd.DataFrame, random_walk_test:pd.DataFrame, train_path:str, valid_path:str, test_path:str, regen:bool):
    if regen in ['all','rw','total']:
        random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]
        random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_valid['random_walk_seq'])]
        # random_walk_valid = random_walk_valid[~random_walk_valid['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]

        random_walk_train.reset_index(drop=True, inplace=True); random_walk_valid.reset_index(drop=True, inplace=True); random_walk_test.reset_index(drop=True, inplace=True)

        random_walk_train.to_csv(train_path, index=False)
        random_walk_valid.to_csv(valid_path, index=False)
        random_walk_test.to_csv(test_path, index=False)

        print(f"rw_train len : {len(random_walk_train)} / rw_valid len : {len(random_walk_valid)} / rw_test len : {len(random_walk_test)}")

    return random_walk_train, random_walk_valid, random_walk_test

def union_user_item_dict(test_dict, valid_dict):
    for k,v in valid_dict.items():
        test_dict[k] = list(set(v).union(set(test_dict.get(k,[]))))
    return test_dict   
    
def generate_input_sequence_data(data_path, user_df, rating_df, seed, split, random_walk_len, item_per_user, augs, regen, used_pairs:dict={}):

    item_seq_len = random_walk_len*item_per_user
    # test set augmentation 여부 확인
    if split=='train':
        total_path = data_path + f"/new_sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_{split}_{augs}.pkl"
    else:
        total_path = data_path + f"/new_sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_{split}.pkl"

    # total_df 재생성 여부 확인
    if os.path.isfile(total_path)&(regen=='no'):
        print(f"{split} total df(input_sequence_data) already exists!")
        print(total_path)
        print(f"Loading {split} total df(input_sequence_data)...")
        total_df = pd.read_pickle(total_path)
        print(f"total df dir : {total_path}")
        return total_df, used_pairs
    
    else:
        print(f"{split} total df(input_sequence_data) doesn't exist!")
        print(f"Creating {split} total df(input_sequence_data)...")
        
        if split=='train':
            used_pairs = {user:set() for user in rating_df.user_id.unique()}
        
        rating_matrix = sparse.load_npz(os.path.join(data_path, 'rating_matrix.npz'))
        user_degree_dic = rating_df.groupby('user_id')['user_degree'].unique().map(lambda x:x.item()).to_dict()
        product_degree_dic = rating_df.groupby('product_id')['product_degree'].unique().map(lambda x:int(x[0])).to_dict()

        interacted_items = rating_df.groupby('user_id')['product_id'].unique().to_dict()
        filtered_items = {user:list(set(items)-used_pairs[user]) for user,items in interacted_items.items()}
        
        def str_to_list(x):
            if type(x)==str:
                return literal_eval(x)
            else:
                return x 
            
        def slice_and_pad_list(input_list, slice_length):
            num_slices = math.ceil(len(input_list) / slice_length)
            input_list += [0] * (slice_length * num_slices - len(input_list))
            result_list = [input_list[i:i + slice_length] for i in range(0, len(input_list), slice_length)]

            return result_list
        
        def map_user_items(user_sequence):
            item_sequence = []
            for user in user_sequence:
                items = filtered_items[user]
                if len(items)>item_per_user:
                    selected = np.random.choice(items, item_per_user, replace=False).tolist()
                elif len(items)==0:
                    selected = [0]
                else:
                    selected = items
                    
                used_pairs[user] = used_pairs.get(user, set()).union(set(selected))
                item_sequence.extend(selected)
            
            return item_sequence
                    
        # str type으로 저장된 데이터 list로 변환 
        tqdm.pandas()
        total_df = pd.DataFrame()
        print("Processing Default information ...")
        total_df['user_sequences'] = user_df.apply(lambda x: str_to_list(x['random_walk_seq']), axis=1)
        total_df['user_degree'] = user_df.apply(lambda x: str_to_list(x['degree']), axis=1)
        total_df['item_sequences'] = total_df['user_sequences'].progress_map(map_user_items) # item mapping + 이전 split에서 사용한 pair 제거
        total_df['item_degree'] = total_df['item_sequences'].progress_map(lambda seq:list(map(lambda x:product_degree_dic.get(x,0), seq)))
        
        total_df['user_id'] = total_df['user_sequences'].progress_map(lambda x:x[0])
        total_df['anchor_degree'] = total_df['user_id'].progress_map(lambda x:user_degree_dic[x])
        total_df['anchor_items'] = total_df['user_id'].progress_map(lambda x:filtered_items[x] if len(filtered_items[x])>0 else [0])
        total_df['anchor_item_degree'] = total_df['anchor_items'].progress_map(lambda seq:list(map(lambda x:product_degree_dic.get(x,0), seq)))

        # slice and pad
        print("Processing Padding & Slicing ...")
        total_df['item_sequences'] = total_df['item_sequences'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df['item_degree'] = total_df['item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df = total_df.explode(['item_sequences','item_degree'])
        total_df['anchor_items'] = total_df['anchor_items'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df['anchor_item_degree'] = total_df['anchor_item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df = total_df.explode(['anchor_items','anchor_item_degree'])
        
        # rating matrix
        total_df['item_rating'] = total_df.progress_apply(lambda x:torch.FloatTensor(rating_matrix[x['user_sequences'],:][:,x['item_sequences']].toarray()), axis=1)
        total_df['anchor_ratings'] = total_df.progress_apply(lambda x:torch.FloatTensor(rating_matrix[x['user_id'],:][:,x['anchor_items']].toarray()), axis=1)        
        
        total_df = total_df[['user_id','user_sequences','user_degree','item_sequences','item_degree','item_rating','anchor_degree','anchor_items','anchor_ratings','anchor_item_degree']]

        with open(total_path, "wb") as file:
            pickle.dump(total_df, file)

        print(f"total df dir : {total_path}")
        print(f"# of total {split} : {len(total_df)}")    
        
    return total_df, used_pairs