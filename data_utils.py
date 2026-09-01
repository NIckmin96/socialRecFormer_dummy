import math
import os
import sys
import json
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

# --- common benchmark preprocessing (single source of truth for filter/remap/split) ---
# lives in the sibling social_rec/ tree so every baseline shares it.
_COMMON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'social_rec'))
if _COMMON_ROOT not in sys.path:
    sys.path.insert(0, _COMMON_ROOT)
from common_preprocess.common import (  # noqa: E402
    load_interactions as _common_interactions,
    load_social as _common_social,
    load_maps as _common_maps,
    load_splits as _common_splits,
)

# 최초 한번만 실행
def mat_to_csv(data_path:str, regen=False):
    # A-BENCH: filtering / id-remap now come from common_preprocess (shared across every
    # baseline). This function only re-materialises SoFT's local artifacts
    # (rating.csv with degree columns, trustnetwork.csv, rating_matrix.npz) from them.
    dataset = os.path.basename(os.path.normpath(data_path))
    rating_path = os.path.join(data_path,'rating.csv')
    trust_path = os.path.join(data_path,'trustnetwork.csv')
    if os.path.isfile(rating_path) & os.path.isfile(trust_path) & (regen!='all'):
        rating_df = pd.read_csv(rating_path)
        trust_df = pd.read_csv(trust_path)

    else:
        print("Creating rating_df, trust_df from common_preprocess artifacts...")
        inter = _common_interactions(dataset)   # user_id,item_id,category_id,rating,timestamp (1..N)
        social = _common_social(dataset)        # user_id_1,user_id_2 (directed, full edge set)
        user_map, item_map = _common_maps(dataset)

        rating_df = (inter[['user_id', 'item_id', 'rating']]
                     .rename(columns={'item_id': 'product_id'})
                     .reset_index(drop=True))
        trust_df = social[['user_id_1', 'user_id_2']].reset_index(drop=True)

        # A-4: persist id mappings (original id -> re-indexed 1..N) for error analysis.
        with open(os.path.join(data_path, 'user_map.json'), 'w') as f:
            json.dump({str(int(k)): int(v) for k, v in user_map.items()}, f)
        with open(os.path.join(data_path, 'item_map.json'), 'w') as f:
            json.dump({str(int(k)): int(v) for k, v in item_map.items()}, f)
        rating_df = add_degree(rating_df, trust_df)

        # 전체 user-item rating 정보를 담은 rating matrix 생성
        n_users = int(rating_df['user_id'].max())+1
        n_items = int(rating_df['product_id'].max())+1
        
        rows = rating_df['user_id'].to_numpy(dtype=np.int32)
        cols = rating_df['product_id'].to_numpy(dtype=np.int32)
        vals = rating_df['rating'].to_numpy(dtype=np.uint8)
        
        rating_matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.uint8).tocsr()
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

# NOTE: the old reset_and_filter_data() (user/item filtering + 1..N re-index) now lives in
# social_rec/common_preprocess/build_common.py::filter_and_remap and is shared by every
# baseline. mat_to_csv() above consumes its output.

def add_degree(rating_df, trust_df):
    rating_df = rating_df.copy()
    
    product_degree = rating_df.groupby('product_id')['user_id'].nunique()
    rating_df['product_degree'] = rating_df['product_id'].map(product_degree).astype('int32')
    
    edges = trust_df[['user_id_1','user_id_2']].copy()
    u = edges[['user_id_1','user_id_2']].min(axis=1)
    v = edges[['user_id_1','user_id_2']].max(axis=1)
    undirected_edges = pd.DataFrame({'u':u, 'v':v}).drop_duplicates()
    
    user_degree = pd.concat([undirected_edges['u'], undirected_edges['v']], ignore_index=True).value_counts(sort=False)
    
    rating_df['user_degree'] = rating_df['user_id'].map(user_degree).fillna(0).astype('int32')
    
    # rating_g = nx.from_pandas_edgelist(rating_df, 'user_id', 'product_id')
    # social_g = nx.from_pandas_edgelist(trust_df, 'user_id_1', 'user_id_2')
    # social_degree = {user:social_g.degree(user) for user in social_g.nodes()}
    
    # rating_df['product_degree'] = rating_df['product_id'].map(lambda x:len(list(rating_g.neighbors(x))))
    # rating_df['user_degree'] = rating_df['user_id'].map(lambda x:social_degree.get(x,0))
    return rating_df


def shuffle_and_split_dataset(data_path:str, df_len, test, seed, regen):
    # 8:1:1 split. `test` is the total held-out ratio (default 0.2); it is split
    # in half into valid / test, so test=0.2 -> train 80% / valid 10% / test 10%.
    dataset = os.path.basename(os.path.normpath(data_path))
    train_path = os.path.join(data_path, f'rating_train_seed_{seed}.csv')
    valid_path = os.path.join(data_path, f'rating_valid_seed_{seed}.csv')
    test_path = os.path.join(data_path, f'rating_test_seed_{seed}.csv')

    if os.path.isfile(train_path) & os.path.isfile(valid_path) & os.path.isfile(test_path) & (regen != 'all'):
        print("Loading Rating split sets...")
        rating_train_set = pd.read_csv(train_path)
        rating_valid_set = pd.read_csv(valid_path)
        rating_test_set = pd.read_csv(test_path)

    else:
        # A-BENCH: the 8:1:1 interaction split is produced once by common_preprocess
        # (byte-identical to the previous in-line shuffle). Here we just load it and
        # re-attach SoFT's degree columns so the downstream schema is unchanged.
        splits = _common_splits(dataset, seed)
        deg = pd.read_csv(os.path.join(data_path, 'rating.csv'))
        deg = deg[['user_id', 'product_id', 'product_degree', 'user_degree']] \
            .drop_duplicates(['user_id', 'product_id'])

        def _prep(df):
            d = (df[['user_id', 'item_id', 'rating']]
                 .rename(columns={'item_id': 'product_id'})
                 .reset_index(drop=True))
            return d.merge(deg, on=['user_id', 'product_id'], how='left')

        rating_test_set = _prep(splits['test'])
        rating_valid_set = _prep(splits['valid'])
        rating_train_set = _prep(splits['train'])
        rating_test_set.to_csv(test_path, index=False)
        rating_valid_set.to_csv(valid_path, index=False)
        rating_train_set.to_csv(train_path, index=False)

    return rating_train_set, rating_valid_set, rating_test_set

def generate_social_dataset(data_path, split, rating_split, trust_df, seed, regen):
    """
    Generate social graph from train/test/validation dataset
    """
    social_file = os.path.join(data_path, f'trustnetwork_{split}_seed_{seed}.csv')
    rating_file = os.path.join(data_path, f'rating_{split}_seed_{seed}.csv')
    if (not os.path.isfile(social_file)) or (not os.path.isfile(rating_file)) or (regen=='all'):
        print(f"Creating Social {split} split sets...\n")
        users = set(rating_split['user_id'])
        social_split = trust_df[(trust_df['user_id_1'].isin(users)) & (trust_df['user_id_2'].isin(users))]

        # save
        social_split.to_csv(social_file, index=False)
        rating_split.to_csv(rating_file, index=False)
    else:
        print("Loading Social split sets...\n")
        social_split = pd.read_csv(social_file)
        rating_split = pd.read_csv(rating_file)
    
    return social_split, rating_split

def generate_social_random_walk_sequence(data_path, rating_split, social_split, walk_length, augs, data_split_seed, split, regen, user_degree_dic=None):
    # A-3: `social_split` is now the FULL shared trust frame (same graph for every split).
    #   - adjacency is built over ALL nodes so a walk can traverse any user
    #   - walks only START from users that are targets in this split (rating_split)
    # The social graph carries no labels, so sharing it leaks nothing; it only makes the
    # eval-time walks as dense as the train-time ones and keeps user degree consistent.
    split_users = set(rating_split['user_id'].unique())
    all_graph_users = pd.unique(pd.concat([social_split['user_id_1'], social_split['user_id_2']], ignore_index=True))
    anchor_count = sum(1 for u in all_graph_users if u in split_users)
    num_anchors = anchor_count*augs

    # save dir 지정
    # all, rw, total
    file_path = os.path.join(data_path, f"rw_rating_length_{walk_length}_{num_anchors}_split_{split}_seed_{data_split_seed}.csv")
    print("Random Walk file dir :", file_path)
    # 이미 random walk 존재하는 경우 return
    if os.path.isfile(file_path) and (regen in ['no','total','train']):
        print(f"Loading {split} random walk sequence file...")
        df = pd.read_csv(file_path)
    # 새로 생성 or regen
    else:
        social_graph = nx.from_pandas_edgelist(social_split, source='user_id_1', target='user_id_2')
        all_nodes = list(social_graph.nodes())
        anchor_nodes = sorted(u for u in all_nodes if u in split_users)
        total_steps = len(anchor_nodes)*augs
        # Cache adjacency over the whole graph (walks pass through non-anchor nodes too).
        adj = {node: tuple(social_graph.neighbors(node)) for node in all_nodes}
        if user_degree_dic is None:
            user_degree_dic = rating_split[['user_id','user_degree']].drop_duplicates('user_id').set_index('user_id')['user_degree'].to_dict()
        anchor_seq_degree = []
        # seq_set = set()
        print(f"{split} random walk sequence file doesn't exist!")
        print(f"Creating {split} random walk sequence...")
        with tqdm(total=total_steps, desc="Generating random walk sequence...") as pbar:
            for _ in range(augs):
                for node in anchor_nodes:
                    seqs = [node]
                    seen = {node} # 중복 체크를 위한 set
                    wl = 1
                    thres=0
                    while wl<walk_length:
                        # 처음 : random next node 추출 후, append
                        if wl == 1:
                            next_node = find_next_node(adj, previous_node=None, current_node=node)
                        # 처음이 아닌 경우
                        else:
                            next_node = find_next_node(adj, previous_node=seqs[-2], current_node=seqs[-1])
                            # 중복 노드가 존재하는 경우, 제외하고 random choice
                            if next_node in seen:
                                if thres < 5:
                                    thres += 1
                                    continue
                                else:
                                    next_node = 0
                                    # available = set(social_graph.nodes())-set(seqs)
                                    # if available:
                                    #     next_node = np.random.choice(list(available))

                        # next node 추가 후, walk length 하나 올림
                        seqs.append(next_node)
                        seen.add(next_node)
                        wl += 1
                
                    # anchor_cnt[node]+=1
                    degrees = [0 if node==0 else user_degree_dic.get(node, 0) for node in seqs]
                    anchor_seq_degree.append([node,seqs,degrees])
                    pbar.update(1)

        # revised
        df = pd.DataFrame(anchor_seq_degree,columns=['user_id','random_walk_seq','degree'])
        df = df.sort_values(by=['user_id'])
        df = df.reset_index(drop=True)
        print(f"split : {split} / len : {len(df)}")
        print("\n")
        df.to_csv(file_path, index=False)

    return df


def find_next_node(adj, previous_node, current_node): 
    '''
    - previosu_node가 None인(sequence의 길이가 1인) 경우에, current_node의 neighbor node를 후보로 삼고 동일하게 1/n의 확률로 next_node를 random select
    - 그렇지 않은 경우, current_node의 neighbor에서 previous node를 제외한 node를 neighbor로 삼고,
        - neighbor node가 존재하지 않는 경우에, 전체 graph에서 random한 node를 뽑아서 다음 node로 삼음(?) => 0으로 padding하도록 수정
        - current_node가 0(sequence padding이 사작)된 경우에, 계속 0을 return
        - previous node로 돌아갈 확률 = 1/(n+max(n,2))
        - previous node를 제외한 다른 노드가 선택될 확률 = (1- previous node가 선택될 확률)/(n)
    '''
    if previous_node is not None:
        if current_node == 0:
            return 0

        neighbors = adj.get(current_node, ())
        candidates = [node for node in neighbors if node != previous_node]
        n = len(candidates)
        if n == 0:
            return 0

        return_prob = 1 / (n + max(n, 2))
        edge_prob = (1 - return_prob) / n
        probs = [edge_prob for _ in range(n)] + [return_prob]
        candidates = candidates + [previous_node]

    else:
        candidates = adj.get(current_node, ())
        n = len(candidates)
        if n == 0:
            return 0

        edge_prob = 1 / n
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
    
def generate_input_sequence_data(data_path, rw_df, rating_split, user_degree_dic, product_degree_dic, rating_matrix, seed, split, random_walk_len, item_per_user, min_item_len, regen, neg, used_pairs=None, context_rating=None, drop_cold_eval=False):

    if used_pairs==None:
        used_pairs={}
    # A-3: encoder context (what the model READS from walk users) is always the TRAIN
    # ratings; only the anchor/target side uses `rating_split`. Passing context_rating=None
    # falls back to the old (leaky) behaviour where both sides use `rating_split`.
    if context_rating is None:
        context_rating = rating_split
        
    item_seq_len = random_walk_len*item_per_user
    # test set augmentation 여부 확인
    if neg==True:
        total_path = data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_{rw_df.shape[0]}_{split}_neg.pkl"
    else:
        total_path = data_path + f"/sequence_data_seed_{seed}_walk_{random_walk_len}_itemlen_{item_seq_len}_{rw_df.shape[0]}_{split}_non_neg.pkl"
    
    # total_df 재생성 여부 확인
    if os.path.isfile(total_path)&(not regen):
        print(f"{split} total df(input_sequence_data) already exists!")
        print(total_path)
        print(f"Loading {split} total df(input_sequence_data)...")
        total_df = pd.read_pickle(total_path)
        print(f"total df dir : {total_path}")
    
    else:
        print(f"{split} total df(input_sequence_data) doesn't exist!")
        print(f"Creating {split} total df(input_sequence_data)...")
        print(f"total df dir : {total_path}")
        
        # if split=='train':
        #     used_pairs = {user:set() for user in rating_split.user_id.unique()}
        
        # filtered_items = {user:list(set(items)-used_pairs.get(user,set())) for user,items in interacted_items.items()}

        # target side (anchor items / which anchors to emit): the split being generated
        interacted_items = rating_split.groupby('user_id')['product_id'].unique().to_dict()
        # encoder-context side (item_sequences / item_rating for every walk user): TRAIN only
        ctx_rating_lookup = {(row.user_id,row.product_id):row.rating for row in context_rating.itertuples(index=False)}
        ctx_interacted_items = context_rating.groupby('user_id')['product_id'].unique().to_dict()
        
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
        
        def pad_list(input_list, pad_length):
            pad = [0]*(pad_length-min(len(input_list), pad_length))
            result_list = input_list + pad
            
            return result_list
        
        # A-3: walk users contribute their TRAIN items only (ctx_interacted_items), so a
        # test sample never sees a walk user's held-out items in the encoder input.
        def map_user_items(user_sequence):
            item_sequence = []
            for user in user_sequence:
                items = ctx_interacted_items.get(user, [0])
                if len(items)>item_per_user:
                    selected = np.random.choice(items, item_per_user, replace=False).tolist()
                elif len(items)==0:
                    selected = [0]
                else:
                    selected  = list(items)
                n = min(item_per_user, item_per_user-len(selected))
                selected.extend([0]*n)
                item_sequence.extend(selected)
            
            return item_sequence
        
        def get_ratings(user_seq, item_seq):
            total_ratings = []
            for user in user_seq:
                ratings = []
                for item in item_seq:
                    rating = ctx_rating_lookup.get((user,item),0)   # A-3: train ratings only
                    ratings.append(rating)
                total_ratings.append(ratings)
            total_ratings = torch.FloatTensor(np.stack(total_ratings))
            
            return total_ratings
        
        all_items = np.arange(1, rating_matrix.shape[1], dtype=np.int32)

        def add_negs(user_id, anchor_items):
            anchor_items = list(set(anchor_items) - {0})
            nnz = rating_matrix[user_id].indices
            zero_indices = np.setdiff1d(all_items, nnz, assume_unique=False)

            n_pos = len(anchor_items)
            n_neg = max(10 - n_pos, n_pos)

            if n_pos + n_neg > item_seq_len:
                if n_pos == n_neg:
                    keep_pos = min(n_pos, item_seq_len // 2)
                    if keep_pos > 0:
                        anchor_items = np.random.choice(anchor_items, size=keep_pos, replace=False).tolist()
                    else:
                        anchor_items = []
                    n_neg = item_seq_len - len(anchor_items)
                else:
                    n_neg = n_pos + n_neg - item_seq_len

            n_neg = min(max(n_neg, 0), len(zero_indices))
            if n_neg > 0:
                neg_samples = np.random.choice(zero_indices, size=n_neg, replace=False).tolist()
            else:
                neg_samples = []

            anchor_items.extend(neg_samples)
            anchor_items = anchor_items[:item_seq_len]

            return {'anchor_items': anchor_items,
                    'neg_samples': neg_samples}
                    
        # str type으로 저장된 데이터 list로 변환 
        tqdm.pandas()
        total_df = pd.DataFrame()
        print("Processing Default information ...")
        total_df['user_sequences'] = rw_df['random_walk_seq'].map(str_to_list)
        # total_df['user_sequences'] = rw_df.apply(lambda x: str_to_list(x['random_walk_seq']), axis=1)
        total_df['user_degree'] = rw_df['degree'].map(str_to_list)
        # total_df['user_degree'] = rw_df.apply(lambda x: str_to_list(x['degree']), axis=1)
        total_df['item_sequences'] = total_df['user_sequences'].progress_map(map_user_items) # item mapping + 이전 split에서 사용한 pair 제거
        total_df['item_degree'] = total_df['item_sequences'].progress_map(lambda seq:list(map(lambda x:product_degree_dic.get(x,0), seq)))
        
        # total df
        print("Processing Total DF...")
        # slice and pad
        item_len = total_df['item_sequences'].apply(len).max()
        total_df['item_sequences'] = total_df['item_sequences'].progress_map(lambda x:slice_and_pad_list(x,item_len))
        total_df['item_degree'] = total_df['item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_len))
        total_df = total_df.explode(['item_sequences','item_degree'])
        # rating matrix
        # total_df['item_rating'] = total_df.progress_apply(lambda x:torch.FloatTensor(rating_matrix[x['user_sequences'],:][:,x['item_sequences']].toarray()), axis=1)
        pairs = zip(total_df['user_sequences'].tolist(), total_df['item_sequences'].tolist())
        total_df['item_rating'] = [get_ratings(u_seq, i_seq) for u_seq, i_seq in tqdm(pairs, total=len(total_df), desc="item_rating")]
        # total_df['item_rating'] = total_df.progress_apply(lambda x:get_ratings(x['user_sequences'], x['item_sequences']), axis=1)
        # total_df['anchor_user'] = total_df['user_sequences'].progress_map(lambda x:x[0])
        # total_df['anchor_degree'] = total_df['anchor_user'].progress_map(lambda x:user_degree_dic[x])
        # total_df['anchor_items'] = total_df['anchor_user'].progress_map(lambda x:interacted_items[x])
        total_df['anchor_user'] = total_df['user_sequences'].str[0]
        total_df['anchor_degree'] = total_df['anchor_user'].map(user_degree_dic)
        total_df['anchor_items'] = total_df['anchor_user'].map(lambda user: list(interacted_items.get(user, [])))
        total_df = total_df[total_df['anchor_items'].map(len) > 0].reset_index(drop=True)

        # Cold-start guard: on non-train splits, optionally drop anchors that have NO
        # training interactions. Their encoder context (ctx_interacted_items) is empty, so
        # the model can only fall back to walk-neighbour signal -- not a fair rating target.
        # Controlled by --drop_cold_eval; default off so existing numbers are unchanged.
        if drop_cold_eval and split != 'train':
            warm_users = set(ctx_interacted_items.keys())
            n_before = len(total_df)
            total_df = total_df[total_df['anchor_user'].isin(warm_users)].reset_index(drop=True)
            print(f"[cold-start] split='{split}': dropped {n_before - len(total_df)} cold anchors "
                  f"(0 train interactions); {len(total_df)} warm anchors kept")

        print("1. anchor item len :", total_df['anchor_items'].apply(len).max(), total_df['anchor_items'].apply(len).min())
        # train/valid/test 중 Minimum보다 큰 경우, random sample [tmp]
        total_df['anchor_items'] = total_df['anchor_items'].progress_map(lambda x:np.random.choice(x, min(len(x),min_item_len), replace=False).tolist())
        print("2. anchor item len :",total_df['anchor_items'].apply(len).max(), total_df['anchor_items'].apply(len).min())
        
        # iteraction item 아무것도 없는 경우 drop
        # total_df.dropna(subset='anchor_items', inplace=True)
        # negative sampling
        if neg:
            print("Negative Sampling...")
            neg_pairs = zip(total_df['anchor_user'].tolist(), total_df['anchor_items'].tolist())
            neg_out = [add_negs(u, items) for u, items in tqdm(neg_pairs, total=len(total_df), desc="neg_sampling")]
            neg_df = pd.DataFrame(neg_out)  # add_negs가 dict 반환하도록 바꾸면 더 깔끔
            total_df[['anchor_items', 'neg_samples']] = neg_df[['anchor_items', 'neg_samples']]
            # total_df[['anchor_items','neg_samples']] = total_df[['anchor_user','anchor_items']].progress_apply(lambda x:add_negs(x['anchor_user'],x['anchor_items']), axis=1)
        print("3. anchor item len :",total_df['anchor_items'].apply(len).max(), total_df['anchor_items'].apply(len).min())
        total_df['anchor_item_degree'] = total_df['anchor_items'].progress_map(lambda seq:list(map(lambda x:product_degree_dic.get(x,0), seq)))
        # slice and pad
        # total_df['anchor_items'] = total_df['anchor_items'].progress_map(lambda x:slice_and_pad_list(x,item_len))
        # total_df['anchor_item_degree'] = total_df['anchor_item_degree'].progress_map(lambda x:slice_and_pad_list(x,item_len))
        # total_df['anchor_items'] = total_df['anchor_items'].progress_map(lambda x:slice_and_pad_list(x,min_item_len))
        # total_df['anchor_item_degree'] = total_df['anchor_item_degree'].progress_map(lambda x:slice_and_pad_list(x,min_item_len))
        total_df['anchor_items'] = total_df['anchor_items'].progress_map(lambda x:pad_list(x[:item_seq_len], item_seq_len))
        total_df['anchor_item_degree'] = total_df['anchor_item_degree'].progress_map(lambda x:pad_list(x[:item_seq_len], item_seq_len))
        print("4. anchor item len :",total_df['anchor_items'].apply(len).max(), total_df['anchor_items'].apply(len).min())
        # total_df = total_df.explode(['anchor_items','anchor_item_degree'])
        # rating matrix
        # total_df['anchor_ratings'] = total_df.progress_apply(lambda x:torch.FloatTensor(rating_matrix[x['anchor_user'],:][:,x['anchor_items']].toarray()), axis=1)        
        a_pairs = zip(total_df['anchor_user'].tolist(), total_df['anchor_items'].tolist())
        total_df['anchor_ratings'] = [torch.FloatTensor(rating_matrix[u, :][:, items].toarray())
            for u, items in tqdm(a_pairs, total=len(total_df), desc="anchor_ratings")]
        
        if neg:
            total_df = total_df[['user_sequences','user_degree','item_sequences','item_degree','item_rating',
                'anchor_user','anchor_degree','anchor_items','anchor_ratings','anchor_item_degree','neg_samples']]
        else:
            total_df = total_df[['user_sequences','user_degree','item_sequences','item_degree','item_rating',
                                 'anchor_user','anchor_degree','anchor_items','anchor_ratings','anchor_item_degree']]

        with open(total_path, "wb") as file:
            pickle.dump(total_df, file)

        print(f"# of total {split} : {len(total_df)}")
    
    return total_df, used_pairs

def get_used_pairs(total_df):
    all_pairs = []
    for users, items, ratings in zip(total_df['user_sequences'], total_df['item_sequences'], total_df['item_rating']):
        users = np.array(users)
        items = np.array(items)
        
        nz_u, nz_i = torch.nonzero(ratings, as_tuple=True)
        pair_u = users[nz_u.numpy()]
        pair_i = items[nz_i.numpy()]
        
        all_pairs.extend(zip(pair_u, pair_i))
        
    return all_pairs
