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
def mat_to_csv(data_path:str, regenerate=False):
    rating_path = os.path.join(data_path,'rating.csv')
    trust_path = os.path.join(data_path,'trustnetwork.csv')
    if os.path.isfile(rating_path) & os.path.isfile(trust_path) & (not regenerate):
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
            
        rating_df, trust_df = reset_and_filter_data(rating_df, trust_df)

        # 전체 user-item rating 정보를 담은 rating matrix 생성
        rating_matrix = sparse.lil_matrix((max(rating_df['user_id'].unique())+1, max(rating_df['product_id'].unique())+1), dtype=np.ubyte)

        for index in rating_df.index:
            rating_matrix[rating_df['user_id'][index], rating_df['product_id'][index]] = rating_df['rating'][index]
            
        rating_matrix = rating_matrix.tocsr()
        sparse.save_npz(os.path.join(data_path, 'rating_matrix.npz'), rating_matrix) # csr matrix 형태로 저장
        # rating_matrix = rating_matrix.toarray()
        # np.save(data_path + '/rating_matrix.npy', rating_matrix)

        rating_df.to_csv(data_path + '/rating.csv', index=False)
        trust_df.to_csv(data_path + '/trustnetwork.csv', index=False)
    
    # data statistics
    print(f"***** Dataset Statistics *****")
    print(f"# of users : {max(rating_df.user_id.max(), trust_df.user_id_1.max(), trust_df.user_id_2.max())+1}")
    print(f"# of users in rating df : {rating_df.user_id.nunique()}")
    assert rating_df.product_id.nunique()==rating_df.product_id.max()
    print(f"# of items : {rating_df.product_id.nunique()}")
    print(f"# of interactions : {rating_df.shape[0]}")
    print(f"# of Social Links : {trust_df.shape[0]}")

    return rating_df, trust_df

def reset_and_filter_data(rating_df:pd.DataFrame, trust_df:pd.DataFrame) -> pd.DataFrame:   
    total_users = set(trust_df.user_id_1.unique()).union(set(trust_df.user_id_2.unique())).intersection(set(rating_df.user_id.unique()))
    rating_df = rating_df[rating_df.user_id.isin(total_users)]
    trust_df = trust_df[trust_df.user_id_1.isin(total_users)|trust_df.user_id_2.isin(total_users)]
    
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

def shuffle_and_split_dataset(data_path:str, test=0.2, seed=42, regenerate=False):
    
    train_path = os.path.join(data_path, f'rating_train_seed_{seed}.csv')
    valid_path = os.path.join(data_path, f'rating_valid_seed_{seed}.csv')
    test_path = os.path.join(data_path, f'rating_test_seed_{seed}.csv')

    if os.path.isfile(train_path) & os.path.isfile(test_path) & (not regenerate):
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
        
        rating_test_set = split_rating_df.iloc[:num_test//2]
        rating_valid_set = split_rating_df.iloc[num_test//2:num_test]
        rating_train_set = split_rating_df.iloc[num_test:]

        rating_test_set.to_csv(data_path + f'/rating_test_seed_{seed}.csv', index=False)
        rating_valid_set.to_csv(data_path + f'/rating_valid_seed_{seed}.csv', index=False)
        rating_train_set.to_csv(data_path + f'/rating_train_seed_{seed}.csv', index=False)
    
    print(f"data split finished, seed: {seed}\n")
    
    return rating_train_set, rating_valid_set, rating_test_set

def generate_social_dataset(data_path:str, split:str, rating_split:pd.DataFrame, trust_df, seed:int=42, regenerate=False):
    """
    Generate social graph from train/test/validation dataset
    """
    social_file = os.path.join(data_path, f'trustnetwork_{split}_seed_{seed}.csv')
    rating_file = os.path.join(data_path, f'rating_{split}_seed_{seed}.csv')
    if (not os.path.isfile(social_file)) or regenerate:
        print(f"Creating Social {split} split sets...\n")
        # trust_df = pd.read_csv(data_path + '/trustnetwork_org.csv', index_col=[]) # social interaction
        users = rating_split['user_id'].unique()            
        social_split = trust_df[(trust_df['user_id_1'].isin(users)) | (trust_df['user_id_2'].isin(users))]

        # save
        social_split.to_csv(social_file, index=False)
        rating_split.to_csv(rating_file, index=False)
    else:
        print("Loading Social split sets...\n")
        social_split = pd.read_csv(social_file)
        rating_split = pd.read_csv(rating_file)
    
    return social_split, rating_split

def generate_user_degree_table(data_path:str, trust_split, split:str='train', seed:int=42, regenerate=False) -> pd.DataFrame:
    
    user_degree_dir = data_path + f'/degree_table_social_{split}_seed_{seed}.csv'
    if os.path.isfile(user_degree_dir) and (not regenerate):
        degree_df = pd.read_csv(user_degree_dir)
    else:
        social_graph = nx.from_pandas_edgelist(trust_split, source='user_id_1', target='user_id_2')
        degrees = {node: val for (node, val) in social_graph.degree()}
        degree_df = pd.DataFrame(degrees.items(), columns=['user_id', 'degree'])
        degree_df.sort_values(by='user_id', ascending=True, inplace=True)
        degree_df.to_csv(user_degree_dir, index=False)    

    return degree_df, degree_df['degree'].max()


def generate_item_degree_table(data_path:str, rating_split:pd.DataFrame, split:str, seed:int=42, regenerate=False) -> pd.DataFrame:
    
    item_degree_dir = data_path + f'/degree_table_item_{split}_seed_{seed}.csv'
    
    if os.path.isfile(item_degree_dir) and (not regenerate):
        degree_df = pd.read_csv(item_degree_dir)
        
    else:
        degree_df = rating_split.groupby('product_id')['user_id'].nunique().reset_index()
        degree_df.columns = ['product_id', 'degree']
        degree_df.to_csv(item_degree_dir, index=False)

    return degree_df, degree_df['degree'].max()

def generate_interacted_items_table(data_path:str, rating_split:pd.DataFrame, degree_table:pd.DataFrame, split:str, seed:int=42, regenerate=False) -> pd.DataFrame:
    
    user_item_dir = data_path + f'/user_item_interaction_{split}_seed_{seed}.csv'
    if os.path.isfile(user_item_dir) and (not regenerate):
        user_item_dataframe = pd.read_csv(user_item_dir)
    else:
        print(f"Creating {split} user-item table...")
        degree_table = dict(zip(degree_table['product_id'], degree_table['degree']))    # for degree mapping.

        user_item_dataframe = rating_split.groupby('user_id').agg({'product_id': list, 'rating': list}).reset_index()
        user_item_dataframe['product_degree'] = user_item_dataframe['product_id'].apply(lambda x: [degree_table[id] for id in x])

        
        empty_data = [0, [0 for _ in range(4)], [0 for _ in range(4)], [0 for _ in range(4)]]
        user_item_dataframe.loc[-1] = empty_data
        user_item_dataframe.index = user_item_dataframe.index + 1
        user_item_dataframe.sort_index(inplace=True)
        user_item_dataframe.to_csv(user_item_dir, index=False)

    return user_item_dataframe

def generate_rw_sequence(data_path:str, rating_split:pd.DataFrame, social_split:pd.DataFrame, user_degree:pd.DataFrame, item_degree:pd.DataFrame, walk_length:int, data_split_seed:int, split:str, regen:bool, used_set:set):
    social_graph = nx.from_pandas_edgelist(social_split, source='user_id_1', target='user_id_2')
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
    
    file_path = os.path.join(data_path, f"rw_rating_length_{len(anchor_nodes)}_split_{split}_seed_{data_split_seed}.csv")
    # 이미 random walk 존재하는 경우 return
    if os.path.isfile(file_path)&(not regen):
        print(f"Loading {split} random walk sequence file...")
        df = pd.read_csv(file_path)
    # 새로 생성 or Regenerate
    else:
        rating_graph = nx.Graph()
        for _,r in rating_split.iterrows():
            u,i = r['user_id'], 'i'+str(r['product_id'])
            rating_graph.add_node(u)
            rating_graph.add_node(i)
            rating_graph.add_edge(u,i)
            
        G_lst = [social_graph, rating_graph]
        rating_items = rating_split.product_id.unique()
        
        user_degree_dic = dict(zip(user_degree.user_id, user_degree.degree))
        item_degree_dic = dict(zip(item_degree.product_id, item_degree.degree))
        # random walk sequence
        item_cnt = {i:0 for i in rating_items} # item이 
        
        anchor_seq_degree = []
        print(f"{split} random walk sequence file doesn't exist!")
        print(f"Creating {split} random walk sequence...")
        for node in tqdm(anchor_nodes, desc="Generating random walk sequence..."):
            seqs = [node]
            wl = 1
            while wl<walk_length:
                # 처음 : random next node 추출 후, append
                if wl == 1:
                    social, next_node = find_next_node(social_graph, rating_graph, previous_node=None, current_node=node, social=True)
                # 처음이 아닌 경우
                else:
                    social, next_node = find_next_node(social_graph, rating_graph, previous_node=seqs[-2], current_node=seqs[-1], social=social)
                    # 중복 노드가 존재하는 경우, 제외하고 random choice
                    if next_node in seqs:
                        G = G_lst[social^1]
                        available = set(G.nodes())-set(seqs)
                        if available:
                            next_node = np.random.choice(list(set(G.nodes())-set(seqs)))
                        else:
                            next_node = 0
                            
                
                if not social:
                    s = tuple(seqs[-2:])
                    if s in used_set:
                        print("삐삐빅")
                        continue
                    else:
                        used_set.add(s)
                        
                seqs.append(next_node)
                wl += 1
            
                # revised
                degrees = []
                for node in seqs:
                    if node[0]=='i':
                        node = int(node.split('i')[-1])
                        degrees.append(item_degree_dic.get(node,0))
                    else:
                        degrees.append(user_degree_dic.get(node,0))
                
            anchor_seq_degree.append([node,seqs,degrees])
            
        # revised
        df = pd.DataFrame(anchor_seq_degree,columns=['user_id','random_walk_seq','degree'])
        df = df.sort_values(by=['user_id'])
        df = df.reset_index(drop=True)
        df.to_csv(file_path, index=False)

    return df, file_path, used_set

def find_next_node(social_graph, rating_graph, previous_node, current_node, social): # 확률적으로, anchor node가 동일하다면 중복되는 random walk sequence가 나올수도 있음
    # 문제 : neighbor가 많을 경우에, 이전 노드로 돌아갈 확률이 다른 노드로 갈 확률보다 높아짐 -> 의도된 것?
    # return param을 고정하지않고, neighbor의 개수에 따라 유동적으로 변하는게 합리적임 -> n개의 neighbor가 있으면, x = (1/n)*n + return, 1 = (1/nx)*n + return/x
    input_G = social_graph if social else rating_graph
    
    if current_node!=0:
        neighbors = list(set(input_G.neighbors(current_node))-{previous_node})
    else:
        neighbors = list(input_G.nodes())
    n = len(neighbors)
    
    if n==0:
        return 0

    if previous_node not in [None,0]:
        return_prob = 1/(n+max(n,2))
        edge_prob = (1-return_prob)/n
        # 정규화(sum=1)
        _sum = return_prob+n*edge_prob
        return_prob /= _sum
        edge_prob /= _sum

        candidates = neighbors + [previous_node]
        probs = [edge_prob for _ in range(n)]+[return_prob]
    else:
        edge_prob = 1/n
        candidates = neighbors
        probs = [edge_prob for _ in range(n)]

    selected_node = np.random.choice(candidates, p=probs)
    
    return social^1, selected_node

def remove_duplicated_random_walk_sequence(random_walk_train:pd.DataFrame, random_walk_valid:pd.DataFrame, random_walk_test:pd.DataFrame, train_path:str, valid_path:str, test_path:str, regenerate:bool):
    if regenerate:
        random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]
        random_walk_train = random_walk_train[~random_walk_train['random_walk_seq'].isin(random_walk_valid['random_walk_seq'])]
        random_walk_valid = random_walk_valid[~random_walk_valid['random_walk_seq'].isin(random_walk_test['random_walk_seq'])]

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
    
def generate_input_sequence_data(data_path, user_df:pd.DataFrame, item_df:pd.DataFrame, seed:int, split:str, random_walk_len:int=150, regen:bool=False, test_user_item:dict={}):
    # test set augmentation 여부 확인
    total_path = data_path + f"/new_sequence_data_seed_{seed}_walk_{random_walk_len}_{split}.pkl"

    # total_df 재생성 여부 확인
    if os.path.isfile(total_path)&(not regen):
        print(f"{split} total df(input_sequence_data) already exists!")
        print(total_path)
        print(f"Loading {split} total df(input_sequence_data)...")
        total_df = pd.read_pickle(total_path)
        print(f"total df dir : {total_path}")
        if split in ['valid','test']:
            return total_df, {}
        else:
            return total_df
    
    else:
        print(f"{split} total df(input_sequence_data) doesn't exist!")
        print(f"Creating {split} total df(input_sequence_data)...")
        
        rating_matrix = sparse.load_npz(os.path.join(data_path, 'rating_matrix.npz')) # main으로
        
        # product_id(list) : product_degree(list)
        all_item, all_degree = [],[]
        for item_list, item_degree in zip(item_df['product_id'], item_df['product_degree']):
            all_item.extend(item_list)
            all_degree.extend(item_degree)

        # user_id : product_id(list)
        user_degree_dic = dict(zip(item_df['']))
        user_product_dic = dict(zip(item_df['user_id'], item_df['product_id']))
        user_rating_dic = dict(zip(item_df['user_id'], item_df['rating']))
        product_degree_dic = dict(zip(all_item, all_degree))
        
        
        def str_to_list(x):
            return literal_eval(x) if type(x)==str else x
        
        def user_item(user):
            items = user_product_dic.get(user, [0,0,0,0])
            return items
        
        total_df = pd.DataFrame()
        # batch용
        total_df['rw_sequence'] = user_df['random_walk_seq'].map(str_to_list)
        total_df['degree'] = user_df['degree'].map(str_to_list)
        
        # valid/test용
        total_df['user'] = total_df['rw_sequence'].map(lambda x:x[0])
        total_df['items'] = total_df['user'].map(user_item)
        
        
        
        
        tqdm.pandas()
        
        # item(anchor user에 해당)
        print("Processing Item sequences / Degrees ...")
        total_df['anchor_degree'] = total_df['user_degree'].map(lambda x:x[0])
        # total_df['anchor_items'] = total_df['user_id'].map(user_product_dic)
        total_df['anchor_items'] = total_df['user_id'].map(user_item)
        total_df['anchor_items'] = total_df['anchor_items'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        # total_df['anchor_ratings'] = total_df['user_id'].map(user_rating_dic)
        total_df['anchor_ratings'] = total_df['user_id'].map(user_rating)
        total_df['anchor_ratings'] = total_df['anchor_ratings'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df = total_df.explode(['anchor_items','anchor_ratings'])
        # total_df = total_df.drop_duplicates(subset='user_id',keep='first')
        total_df['anchor_item_degree'] = total_df['anchor_items'].progress_map(lambda seq:list(map(lambda x:product_degree_dic[x], seq)))
        total_df['imp_fdback'] = total_df['anchor_ratings'].map(lambda seq:list(map(lambda x:1 if x>=rating_thres else 0,seq)))
        # item(user sequence에 해당)\
        total_df['item_sequences'] = total_df['user_sequences'].progress_map(lambda seq:list(map(map_user_item, seq)))
        total_df['item_sequences'] = total_df['item_sequences'].progress_map(lambda x:sum(x,start=[])).map(lambda x:list(set(x)-set([0])))
        total_df['item_degree'] = total_df['item_sequences'].progress_map(lambda seq:list(map(lambda x:product_degree_dic[x], seq)))
        # slice item & degree
        print("Processing Item sequences/Degrees Slicing ...")
        total_df['item_sequences'] = total_df['item_sequences'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df['item_degree'] = total_df['item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        total_df = total_df.explode(['item_sequences','item_degree']).reset_index(drop=True)

        # # spd matrix
        # print("Processing Spd Matrix ...")
        # total_df['spd_matrix'] = total_df['user_sequences'].progress_map(lambda x:spd_table[torch.LongTensor(x)-1].T[torch.LongTensor(x)-1])
        
        del product_degree_dic, user_product_dic, user_rating_dic
        
        print("Processing Rating Matrix ...")
        total_df['item_rating'] = total_df.progress_apply(lambda x:torch.LongTensor(rating_matrix[x['user_sequences'],:][:,x['item_sequences']].toarray().astype(int)), axis=1)
        
        del rating_matrix
        
        # dev -> spd_loss 제거
        total_df = total_df[['user_id','user_sequences','user_degree','anchor_degree','anchor_items','anchor_item_degree','imp_fdback','anchor_ratings',
                             'item_sequences','item_degree','item_rating']]

        with open(total_path, "wb") as file:
            pickle.dump(total_df, file)

    print(f"total df dir : {total_path}")
    print(f"# of total {split} : {len(total_df)}")    
    
    if split in ['valid', 'test']:
        return total_df, test_user_product_dict
    else:
        return total_df
    
def generate_ranking_test_data(data_path, rating_test, item_degree_test, user_item_train, user_item_test, user_degree_df, seed, user_seq_len, item_per_user, rating_thres, regenerate):
    file_path = os.path.join(data_path, f'ranking_test_data_seed_{seed}.csv')
    
    if os.path.isfile(file_path) and (not regenerate):
        ranking_test_df = pd.read_pickle(file_path)
    else:
        print("Creating Ranking test dataset...")
        spd_table = torch.from_numpy(np.load(data_path + '/' + 'shortest_path_result.npy')).long()
        item_seq_len = user_seq_len*item_per_user
        # 1. test - train
        ranking_data = pd.DataFrame()
        ranking_data['user_id'] = user_item_test['user_id'].copy()
        ranking_data = ranking_data.iloc[1:,].reset_index(drop=True)
        ranking_data['product_id'] = ranking_data['user_id'].map(lambda x:list(set(*user_item_test[user_item_test['user_id']==x]['product_id'])-set(*user_item_train[user_item_train['user_id']==x]['product_id'])))
        ranking_data['product_id'] = ranking_data['product_id'].map(str_to_list)
        # 2. rating 정보 붙이기
        user_item_rating = rating_test.groupby('user_id').agg({'product_id':list, 'rating':list})
        user_item_rating['dic'] = user_item_rating.apply(lambda x:dict(zip(x['product_id'],x['rating'])), axis=1)
        user_item_rating_dic = dict(zip(user_item_rating.index, user_item_rating['dic']))
        ranking_data['rating'] = ranking_data.apply(lambda x:list(map(lambda y:user_item_rating_dic[x['user_id']][y], x['product_id'])), axis=1)
        ranking_data['rating'] = ranking_data['rating'].map(str_to_list)
        # 3. product degree 정보
        item_degree_dic = dict(zip(item_degree_test['product_id'], item_degree_test['degree']))
        ranking_data['product_degree'] = ranking_data['product_id'].map(lambda seq:list(map(lambda x:item_degree_dic[x], seq)))
        ranking_data['product_degree'] = ranking_data['product_degree'].map(str_to_list)
        # 4. user 정보 붙이기
        user_degree_dict = dict(zip(user_degree_df['user_id'], user_degree_df['degree']))
        ranking_test_df = ranking_data[['user_id','product_id','product_degree','rating']]
        ranking_test_df['user_sequences'] = ranking_test_df['user_id'].progress_map(lambda x:[x]*user_seq_len)
        ranking_test_df['user_sequences'] = ranking_test_df['user_sequences'].map(str_to_list)
        ranking_test_df['user_degree'] = ranking_test_df['user_id'].progress_map(lambda x:[user_degree_dict[x]]*user_seq_len)
        ranking_test_df['user_degree'] = ranking_test_df['user_degree'].map(str_to_list)
        # 5. explicit rating to implicit rating
        ranking_test_df['imp_fdback'] = ranking_test_df['rating'].progress_map(lambda seq:list(map(lambda x:1 if x>=rating_thres else 0, seq)))
        # 6. user distance
        ranking_test_df['spd_matrix'] = ranking_test_df['user_sequences'].progress_map(lambda x:spd_table[torch.LongTensor(x)-1].T[torch.LongTensor(x)-1])
        # 7. columns 순서 재배치 & rename
        ranking_test_df = ranking_test_df.loc[:,['user_id','user_sequences','user_degree','product_id','product_degree','imp_fdback','rating','spd_matrix']]
        ranking_test_df.rename(columns={'product_id':'item_sequences',
                                        'product_degree':'item_degree',
                                        'rating':'item_rating'},
                               inplace=True)
        # padding & slicing
        ranking_test_df['item_sequences'] = ranking_test_df['item_sequences'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        ranking_test_df['item_degree'] = ranking_test_df['item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        ranking_test_df['imp_fdback'] = ranking_test_df['imp_fdback'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        ranking_test_df['item_rating'] = ranking_test_df['item_rating'].progress_map(lambda x:slice_and_pad_list(x,item_seq_len))
        # explode item sequence
        ranking_test_df = ranking_test_df.explode(['item_sequences','item_degree','imp_fdback','item_rating'])
        # keep sequences by 'item_seq_len' and drop leftovers
        ranking_test_df = ranking_test_df.drop_duplicates(subset='user_id',keep='first')
        # drop na(test-train 하면서 결측치가 있는 부분 생길 수 있음)
        ranking_test_df = ranking_test_df.dropna(axis=0, how='any')
        # sort by 'user_id'
        ranking_test_df = ranking_test_df.sort_values(by='user_id')
        ranking_test_df = ranking_test_df.reset_index(drop=True)
        # save
        with open(file_path, "wb") as file:
            pickle.dump(ranking_test_df, file)
        
    return ranking_test_df    

def pad_list(input_list:list, slice_length:int):
        """
        Get list, and slice it by slice length, and pad with 0.
        """
        if len(input_list) < slice_length:
            input_list += [0] * (slice_length - len(input_list))

        return input_list