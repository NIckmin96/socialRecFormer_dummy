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
import time
import pickle
import random
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import multiprocessing as mp
from ast import literal_eval    # convert str type list to original type
from scipy.io import loadmat
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
import torch
from scipy import sparse

# 최초 한번만 실행
def mat_to_csv(data_path:str):
    # rating_df : user-item interaction data
    # trust_df : user간의 Social interaction을 나타내는 데이터
    rating_path = os.path.join(data_path,'rating.csv')
    trust_path = os.path.join(data_path,'trustnetwork.csv')
    if os.path.isfile(rating_path) & os.path.isfile(trust_path):
        rating_df = pd.read_csv(rating_path)
        trust_df = pd.read_csv(trust_path)
        return rating_df, trust_df
    dataset_name = data_path.split('/')[-1]

    # rating_df
    if dataset_name=='ciao':
        rating_file = loadmat(data_path + '/' + 'rating.mat')
        rating_file = rating_file['rating'].astype(np.int64)
        rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness'])
        # drop unused columns (TODO: Maybe used later)
        rating_df.drop(['category_id', 'helpfulness'], axis=1, inplace=True)
    elif dataset_name == 'epinions':
        rating_file = loadmat(data_path + '/' + 'rating_with_timestamp.mat')
        rating_file = rating_file['rating_with_timestamp'].astype(np.int64)
        rating_df = pd.DataFrame(rating_file, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness', 'timestamp'])
        # drop unused columns (TODO: Maybe used later)
        rating_df.drop(['category_id', 'helpfulness', 'timestamp'], axis=1, inplace=True)

    # trust_df        
    trust_file = loadmat(data_path + '/' + 'trustnetwork.mat')
    trust_file = trust_file['trustnetwork'].astype(np.int64)    
    trust_df = pd.DataFrame(trust_file, columns=['user_id_1', 'user_id_2'])

    ### data filtering & id-rearrange ###
    # 1. trust_df(socical interaction) & rating_df(user-item interaction) 공통되는 user만 사용
    # 2. Re-index : user_id를 1부터 순차적으로 rearrange
    rating_df, trust_df = reset_and_filter_data(rating_df, trust_df)

    # 전체 user-item rating 정보를 담은 rating matrix 생성
    rating_matrix = sparse.lil_matrix((max(rating_df['user_id'].unique())+1, max(rating_df['product_id'].unique())+1), dtype=np.uint16)

    for index in rating_df.index:
        rating_matrix[rating_df['user_id'][index], rating_df['product_id'][index]] = rating_df['rating'][index]
    rating_matrix = rating_matrix.toarray()
    np.save(data_path + '/rating_matrix.npy', rating_matrix)

    # 저장되는 .csv 파일은 user filter & id re-arrange가 완료된 .csv 파일
    # 따라서 이 함수가 한번 실행된 이후로는 사용 X.
    rating_df.to_csv(data_path + '/rating.csv', index=False)
    trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)

    print(".mat file converting finished...")
    return rating_df, trust_df

def reset_and_filter_data(rating_df:pd.DataFrame, trust_df:pd.DataFrame) -> pd.DataFrame:
    """
    Remove users not existing in social graph &
    Re-arrange data ids to be increasing by 1.
    
    This function is used in `mat_to_csv()`. 
    (`find_non_existing_user_in_social_graph()` is deprecated & merged into this function.)

    Args:
        rating_df: originally loaded `rating_df` (user-item interaction data)
        trust_df: originaly loaded `trust_df` (social data)
    """
    social_network = nx.from_pandas_edgelist(trust_df, source='user_id_1', target='user_id_2')

    social_ids = []
    for user_id in social_network.nodes():
        social_ids.append(user_id)
    
    user_item_ids = rating_df['user_id'].unique().tolist()

    # Find users not exists in social data
    non_users = np.setxor1d(user_item_ids, social_ids)

    # Remove users not exists in social data
        # Ciao: 7375 user (user-item) ==> 7317 user (social)
        # Epinions: 22164 user (user-item) ==> 18098 user (social)
    rating_df = rating_df[~rating_df['user_id'].isin(non_users.tolist())].copy()
    
    # Generate user id mapping table
    mapping_table_user = {user_id:idx+1 for idx,user_id in enumerate(social_ids)}
    # Generate item id mapping table
    mapping_table_item = {item_id:idx+1 for idx,item_id in enumerate(rating_df['product_id'].unique())}    

    rating_df['user_id']= rating_df['user_id'].map(mapping_table_user)
    rating_df['product_id'] = rating_df['product_id'].map(mapping_table_item)
    trust_df['user_id_1']= trust_df['user_id_1'].map(mapping_table_user)
    trust_df['user_id_2'] = trust_df['user_id_2'].map(mapping_table_user)

    return rating_df, trust_df

def shuffle_and_split_dataset(data_path:str, test=0.2, seed=42, regenerate=False):
    """
    Split rating.csv file into train/valid/test.
    
    Args:
        data_path: Path to dataset (ciao or epinions)
        test: percentage of test & valid dataset (default: 20%)
        seed: random seed (default=42)
    """
    train_path = os.path.join(data_path, f'rating_train_seed_{seed}.csv')
    # valid_path = os.path.join(data_path, f'rating_valid_seed_{seed}.csv')
    test_path = os.path.join(data_path, f'rating_test_seed_{seed}.csv')

    # if os.path.isfile(train_path)&os.path.isfile(valid_path)&os.path.isfile(test_path):
    if os.path.isfile(train_path)&os.path.isfile(test_path) and (not regenerate):
        print("Loading Rating split sets...")
        rating_train_set = pd.read_csv(train_path)
        # rating_valid_set = pd.read_csv(valid_path)
        rating_test_set = pd.read_csv(test_path)
        
    else:
        print("Creating Rating split sets...")
        rating_df = pd.read_csv(data_path + '/rating.csv', index_col=[])
        ### train test split TODO: Change equation for split later on    
        split_rating_df = shuffle(rating_df, random_state=seed)
        num_test = int(len(split_rating_df)*test)
        
        rating_test_set = split_rating_df.iloc[:num_test]
        rating_train_set = split_rating_df.iloc[num_test:]

        # rating_test_set = split_rating_df[split_rating_df['user_id'].isin(test_users)]
        # rating_train_set = split_rating_df[split_rating_df['user_id'].isin(train_users)]

        rating_test_set.to_csv(data_path + f'/rating_test_seed_{seed}.csv', index=False)
        rating_train_set.to_csv(data_path + f'/rating_train_seed_{seed}.csv', index=False)

    print(f"# of train unique users : {rating_train_set.user_id.nunique()} / # of test unique users : {rating_test_set.user_id.nunique()}")
    print(f"data split finished, seed: {seed}")
    
    return rating_train_set, rating_test_set

def generate_social_dataset(data_path:str, split:str, rating_split:pd.DataFrame, seed:int=42, regenerate=False):
    """
    Generate social graph from train/test/validation dataset
    """
    split_file = os.path.join(data_path, f'trustnetwork_{split}_seed_{seed}.csv')
    if not os.path.isfile(split_file) or regenerate:
        trust_dataframe = pd.read_csv(data_path + '/trustnetwork.csv', index_col=[]) # social interaction
        users = rating_split['user_id'].unique()            
        social_split = trust_dataframe[(trust_dataframe['user_id_1'].isin(users)) & (trust_dataframe['user_id_2'].isin(users))]
        # save
        social_split.to_csv(split_file)
    else:
        social_split = pd.read_csv(split_file)
    
    return social_split

def generate_user_degree_table(data_path:str, trust_split, split:str='train', seed:int=42, regenerate=False) -> pd.DataFrame:
    """
    Generate & return degree table from social graph(trustnetwork).

    # user-user network
        # Ciao: 7317 users
        # Epinions: 18098 users

    """
    user_degree_dir = data_path + f'/degree_table_social_{split}_seed_{seed}.csv'
    if os.path.isfile(user_degree_dir) and (not regenerate):
        degree_df = pd.read_csv(user_degree_dir)
    else:
        social_graph = nx.from_pandas_edgelist(trust_split, source='user_id_1', target='user_id_2')
        degrees = {node: val for (node, val) in social_graph.degree()}
        degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])
        degree_df.sort_values(by='user_id', ascending=True, inplace=True)
        degree_df.to_csv(user_degree_dir, index=False)    

    return degree_df


def generate_item_degree_table(data_path:str, rating_split:pd.DataFrame, split:str, seed:int=42, regenerate=False) -> pd.DataFrame:
    """
    Generate & return degree table from user-item graph(rating matrix).

    Ciao: 7375 user // 105114 items ==> 7317 user // 104975 items (after filtered)
    """
    item_degree_dir = data_path + f'/degree_table_item_{split}_seed_{seed}.csv'
    if os.path.isfile(item_degree_dir) and (not regenerate):
        degree_df = pd.read_csv(item_degree_dir)
    else:
        # Since using NetworkX to compute bipartite graph's degree is time-consuming(because graph is too sparse),
        # we just use pandas for simple degree calculation.
        degree_df = rating_split.groupby('product_id')['user_id'].nunique().reset_index()
        degree_df.columns = ['product_id', 'degree']

        degree_df.to_csv(item_degree_dir, index=False)

    return degree_df

def generate_interacted_items_table(data_path:str, rating_split:pd.DataFrame, degree_table:pd.DataFrame, split:str, seed:int=42, regenerate=False) -> pd.DataFrame:
    """
    Generate & return user's interacted items & ratings table from user-item graph(rating matrix)

    Args:
        data_path: path to dataset
        item_length: number of interacted items to fetch
        seed: random seed, used in dataset split
    """
    user_item_dir = data_path + f'/user_item_interaction_{split}_seed_{seed}.csv'
    if os.path.isfile(user_item_dir) and (not regenerate):
        user_item_dataframe = pd.read_csv(user_item_dir)
    else:
        degree_table = dict(zip(degree_table['product_id'], degree_table['degree']))    # for id mapping.

        user_item_dataframe = rating_split.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()
        user_item_dataframe['product_degree'] = user_item_dataframe['product_id'].apply(lambda x: [degree_table[id] for id in x])

        # This is for indexing 0, where random walk sequence has padded with 0.
            # minimum number of interacted item is 4(before dataset splitting), so pad it to 4.
        empty_data = [0, [0 for _ in range(4)], [0 for _ in range(4)], [0 for _ in range(4)]]
        user_item_dataframe.loc[-1] = empty_data
        user_item_dataframe.index = user_item_dataframe.index + 1
        user_item_dataframe.sort_index(inplace=True)
        # user_item_dataframe.to_csv(data_path + '/user_item_interaction.csv', index=False)
        user_item_dataframe.to_csv(user_item_dir, index=False)

    return user_item_dataframe


def generate_social_random_walk_sequence(data_path:str, social_split:pd.DataFrame, user_degree:pd.DataFrame, walk_length:int=5, data_split_seed:int=42, split:str='train', return_params:int=1, train_augs:int=10, test_augs:bool=False, regenerate:bool=False):
    """
    Generate random walk sequence from social graph(trustnetwork).
    Return:
        List containing dictionaries,\n 
        ex) [
            {user_id: [[user, ..., user], [degree, ..., degree]]}, 
            {user_id: [[user, ..., user], [degree, ..., degree]]},
            ...
            ]
    Args:
        data_path: path to data
        num_nodes: number of nodes to generate random walk sequence
        walk_length: length of random walk
        save_flag: save result locally (default=False)
        all_node: get all node's random walk sequence (default=False)
        data_split_seed: random seed, used in dataset split
        seed: random seed, True or False (default=False)
        split: dataset split type (default=train)
        regenerate: to generate random walk sequence once again (default=False)
    """

    social_split = nx.from_pandas_edgelist(social_split, source='user_id_1', target='user_id_2')
    if split=='train':
        anchor_nodes = np.repeat(social_split.nodes(), train_augs)
    elif split=='test':
        if test_augs:
            test_augs = min(train_augs, 3) # test set augmentation은 최대 3배까지
            anchor_nodes = np.repeat(social_split.nodes(), test_augs)
        else:
            anchor_nodes = social_split.nodes()
    else:
        anchor_nodes = social_split.nodes()

    if split=='train':
        file_path = os.path.join(data_path, f"social_user_{len(social_split.nodes())}_rw_length_{walk_length}_rp_{return_params}_split_{split}_seed_{data_split_seed}_{train_augs}times.csv")
    elif split=='valid':
        file_path = os.path.join(data_path, f"social_user_{len(social_split.nodes())}_rw_length_{walk_length}_rp_{return_params}_split_{split}_seed_{data_split_seed}.csv")
    else:
        if test_augs:
            file_path = os.path.join(data_path, f"social_user_{len(social_split.nodes())}_rw_length_{walk_length}_rp_{return_params}_split_{split}_seed_{data_split_seed}_{test_augs}times.csv")
        else:
            file_path = os.path.join(data_path, f"social_user_{len(social_split.nodes())}_rw_length_{walk_length}_rp_{return_params}_split_{split}_seed_{data_split_seed}.csv")

    # dataframe return 시키는 부분
    if os.path.isfile(file_path)&(not regenerate):
        print(f"Loading {split} random walk sequence file...")
        df = pd.read_csv(file_path)
    
    else:
        user_degree_dic = dict(zip(user_degree.user_id, user_degree.degree)) # for revised code(hashing)
        # random walk sequence
        anchor_seq_degree = []
        seq_set = set()
        print(f"{split} random walk sequence file doesn't exist!")
        print(f"Creating {split} random walk sequence...")
        for nodes in tqdm(anchor_nodes, desc="Generating random walk sequence..."):
            while True:
                seqs = [nodes]
                wl = 0
                threshold = 0
                s2 = set()
                while wl < walk_length-1:
                    # 처음 : random next node 추출 후, append
                    if wl == 0:
                        next_node = find_next_node(social_split, previous_node=None, current_node=nodes, RETURN_PARAMS=0.0)
                    # 처음이 아닌 경우
                    else:
                        # 가장 최근 노드가 '0'인 경우 : 다음도 '0'
                        if seqs[-1]==0:
                            next_node = 0
                        # 그렇지 않은 경우 : random node 추출, 이미 추가된 노드이면 threshold올리고 다시 추출, threshold 넘으면 '0' append
                        else:
                            next_node = find_next_node(social_split, previous_node=seqs[-2], current_node=seqs[-1], RETURN_PARAMS=return_params/10)
                            if next_node in seqs:
                                threshold+=1
                                if threshold > 10:
                                    next_node = 0
                                else:
                                    continue
                    # next node 추가 후, walk length 하나 올림
                    seqs.append(next_node)
                    wl += 1
            
                # revised
                degrees = [0 if node==0 else user_degree_dic[node] for node in seqs]
                tmp = [nodes, seqs, degrees]
                # 중복 확인
                s2.add(tuple(seqs))
                if s2&seq_set: # 중복되는 sequence가 있는 경우에 sequence 재생성
                    print("삐삐빅")
                    continue
                else:
                    break
                
            anchor_seq_degree.append([nodes,seqs,degrees])

        # revised
        df = pd.DataFrame(anchor_seq_degree,columns=['user_id','random_walk_seq','degree'])
        df = df.sort_values(by=['user_id'])
        df = df.reset_index(drop=True)
        df.to_csv(file_path, index=False)

    return df, file_path

def find_next_node(input_G, previous_node, current_node, RETURN_PARAMS): # 확률적으로, anchor node가 동일하다면 중복되는 random walk sequence가 나올수도 있음
    """
    input_G의 current_node에서 weight를 고려하여 다음 노드를 선택함. 
    - 이 과정에서 RETURN_params를 고려함. 
    - 이 값은 previous_node로 돌아가는가 돌아가지 않는가를 정하게 됨. 
    """
        
    select_probabilities = {}
    
    for node in input_G.neighbors(current_node):
        if node != previous_node:
            select_probabilities[node] = 1   
        
    select_probabilities_sum = sum(select_probabilities.values())
    select_probabilities = {k: v/select_probabilities_sum*(1-RETURN_PARAMS) for k, v in select_probabilities.items()}
    if previous_node is not None:
        select_probabilities[previous_node]=RETURN_PARAMS # 이 노드는 RETURN_PARAMS에 의해 결정됨. 
    
    if select_probabilities_sum == 0:
        return 0
    
    selected_node = np.random.choice(
        a=[k for k in select_probabilities.keys()],
        p=[v for v in select_probabilities.values()]
    )
    return selected_node

def remove_duplicated_social_random_walk_sequence(data_path:str, random_walk_train:pd.DataFrame, random_walk_test:pd.DataFrame, train_path:str, test_path:str):
    random_walk_train = random_walk_train.drop_duplicates('random_walk_seq')
    random_walk_test = random_walk_test.drop_duplicates('random_walk_seq')
    random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]

    random_walk_train.to_csv(train_path)
    random_walk_test.to_csv(test_path)

    print(f"rw_train len : {len(random_walk_train)} rw_test len : {len(random_walk_test)}")

    return random_walk_train, random_walk_test

def slice_and_pad_list(input_list:list, slice_length:int):
        """
        Get list, and slice it by slice length, and pad with 0.
        """
        num_slices = math.ceil(len(input_list) / slice_length)

        # Pad input list with 0
        input_list += [0] * (slice_length * num_slices - len(input_list))

        # Create sliced & padded list
        result_list = [input_list[i:i + slice_length] for i in range(0, len(input_list), slice_length)]

        return result_list
    
# Load dataset & convert data type
def str_to_list(x):
    if type(x)==str:
        return literal_eval(x)
    else:
        return x 
    
def generate_input_sequence_data(data_path, user_df:dict, item_df:dict, seed:int, split:str, random_walk_len:int=30, item_seq_len:int=100, return_params:int=1, train_augs:int=10, test_augs:bool=False, regenerate:bool=False, test_user_item:dict={}):

    # if os.path.isfie(data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_{split}.pkl"):
    #     print(data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_{split}.pkl"+" file exists")
    #     return
    """
    Prepare data to fed into Dataset class.:

    data_path: path to dataset (/dataset/{ciao,epinions}/)
    seed: random seed, used in dataset split
    split: data split type (train/valid/test)
    random_walk_len: pre-defined random walk sequence's length (used in `generate_social_random_walk_sequence()`)
    item_seq_len: pre-defined interacted item sequence length

    FIXME: 현재는 .csv로 저장 중. 추후 return을 한다면 아래와 같이 return을 할 수 있게 수정?

    Returns:
        user_seq:       고정된 길이의 사용자 랜덤워크 시퀀스, [num_user, seq_length]\n
        user_degree:    랜덤워크 시퀀스에서 출현한 사용자들의 degree 정보, [num_user, seq_length] \n
        item_seq:       랜덤워크 시퀀스에서 출현한 사용자들이 상호작용한 모든 아이템 리스트 \n
        item_rating:    랜덤워크 시퀀스에서 출현한 사용자들이 상호작용한 모든 아이템에 대한 rating 정보가 담긴 matrix \n
        item_degree:    선택된 아이템들의 degree 정보 (해당 아이템과 상호작용한 사용자의 수) \n
        spd_matrix:     현재 user_seq에 해당하는 사용자들의 SPD matrix (사전 생성한 전체 사용자의 SPD table에서 slicing한 matrix)
    """

    spd_path = 'shortest_path_result.npy'
    # test set augmentation 여부 확인
    if split=='train':
        total_path = data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_rp_{return_params}_train_{train_augs}times.pkl"
    elif split=='valid':
        total_path = data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_rp_{return_params}_valid.pkl"
    else:
        if test_augs:
            test_augs = min(train_augs, 3) # test augmentation은 최대 3배까지
            total_path = data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_rp_{return_params}_test_{test_augs}times.pkl"
        else:
            total_path = data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_rp_{return_params}_test.pkl"

    # total_df 재생성 여부 확인
    if os.path.isfile(total_path)&(not regenerate):
        print(f"{split} total df(input_sequence_data) already exists!")
        print(total_path)
        print(f"Loading {split} total df(input_sequence_data)...")
        total_df = pd.read_pickle(total_path)
        if split=='test':
            return total_df, {}
        else:
            return total_df
    
    else:
        print(f"{split} total df(input_sequence_data) doesn't exist!")
        print(f"Creating {split} total df(input_sequence_data)...")
        # Load SPD table => 각 sequence마다 [seq_len_user, seq_len_user] 크기의 SPD matrix를 생성하도록.
        spd_table = torch.from_numpy(np.load(data_path + '/' + spd_path)).long()
        
        # Load rating table => 마찬가지로 각 sequence마다 [seq_len_user, seq_len_item] 크기의 rating matrix를 생성하도록.
        rating_matrix = np.load(data_path + '/rating_matrix.npy')
        
        # str type으로 저장된 데이터 list로 변환
        user_df['random_walk_seq'] = user_df.apply(lambda x: str_to_list(x['random_walk_seq']), axis=1)
        user_df['degree'] = user_df.apply(lambda x: str_to_list(x['degree']), axis=1)

        item_df['product_id'] = item_df.apply(lambda x: str_to_list(x['product_id']), axis=1)
        item_df['rating'] = item_df.apply(lambda x: str_to_list(x['rating']), axis=1)
        item_df['product_degree'] = item_df.apply(lambda x: str_to_list(x['product_degree']), axis=1)
        
        # user_id : product_id(list)
        user_product_dic = dict(zip(item_df['user_id'], item_df['product_id']))
        
        # product_id(list) : product_degree(list)
        all_item, all_degree = [],[]
        for item_list, item_degree in zip(item_df['product_id'], item_df['product_degree']):
            all_item.extend(item_list)
            all_degree.extend(item_degree)

        product_degree_dic = dict(zip(all_item, all_degree))
        
        test_user_product_dict = {}
        # user당 item mapping하는 함수 + Test셋에서 사용한 조합(user-item) train셋에서 중복 제거
        def map_user_item(user):
            items_per_user = item_seq_len//random_walk_len
            items = list(set(user_product_dic[user]) - set(test_user_item.setdefault(user,[])))
            items = list(set(items)-set([0])) # drop duplicates + 0 제외
            if len(items)>items_per_user:
                selected_items = np.random.choice(items, items_per_user, replace=False)
            elif 0<len(items)<=items_per_user:
                selected_items = items
                diff = items_per_user - len(selected_items)
                selected_items.extend(np.random.choice(items, diff))
            else:
                selected_items = []
                
            test_user_product_dict.setdefault(user,[]).extend(selected_items)
            
            return list(selected_items)

        ############################################ NEW ############################################
        # new total df creation method(based on map function)
        tqdm.pandas()

        total_df = pd.DataFrame(columns=['user_id', 'user_sequences', 'user_degree', 'item_sequences', 'item_degree', 'item_rating', 'spd_matrix'])
        # user
        total_df['user_id'] = user_df['user_id']
        total_df['user_sequences'] = user_df['random_walk_seq']
        total_df['user_degree'] = user_df['degree']
        # item
        print("Processing Item sequences / Degrees ...")
        # total_df['item_sequences'] = total_df['user_sequences'].progress_map(lambda seq:list(map(lambda x:user_product_dic[x], seq)))
        total_df['item_sequences'] = total_df['user_sequences'].progress_map(lambda seq:list(map(map_user_item, seq)))
        # print(total_df['item_sequences'][0])
        total_df['item_sequences'] = total_df['item_sequences'].progress_map(lambda x:sum(x,start=[])).map(lambda x:list(set(x)-set([0])))
        total_df['item_degree'] = total_df['item_sequences'].progress_map(lambda seq:list(map(lambda x:product_degree_dic[x], seq)))
        # slice item & degree
        print("Processing Item sequences/Degrees Slicing ...")
        total_df['item_sequences'] = total_df['item_sequences'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df['item_degree'] = total_df['item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df = total_df.explode(['item_sequences','item_degree']).reset_index(drop=True)
        # print(total_df.shape)
        # spd matrix
        print("Processing Spd Matrix ...")
        total_df['spd_matrix'] = total_df['user_sequences'].progress_map(lambda x:spd_table[torch.LongTensor(x)-1].T[torch.LongTensor(x)-1])
        # rating matrix(user-item)
        print("Processing Rating Matrix ...")
        total_df['item_rating'] = total_df.progress_apply(lambda x:torch.LongTensor(rating_matrix[x['user_sequences'],:][:,x['item_sequences']].astype(int)), axis=1)

        with open(total_path, "wb") as file:
            pickle.dump(total_df, file)

    print(f"# of total {split} : {len(total_df)}")    
    
    if split=='test':
        return total_df, test_user_product_dict
    else:
        return total_df

def pad_list(input_list:list, slice_length:int):
        """
        Get list, and slice it by slice length, and pad with 0.
        """
        if len(input_list) < slice_length:
            input_list += [0] * (slice_length - len(input_list))

        return input_list