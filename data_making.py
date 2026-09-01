import os
import scipy.sparse as sparse
import data_utils as utils

class DatasetMaking:
    def __init__(self, args):
        data_path = os.getcwd() + '/dataset/' + args.dataset
        # create fundamental dataframe (Rating / Social)
        self.rating_df, self.trust_df = utils.mat_to_csv(data_path, args.regen)
        self.rating_matrix = sparse.load_npz(os.path.join(data_path, 'rating_matrix.npz'))
        self.user_degree_dic = self.rating_df.groupby('user_id')['user_degree'].unique().map(lambda x:x.item()).to_dict()
        self.product_degree_dic = self.rating_df.groupby('product_id')['product_degree'].unique().map(lambda x:int(x[0])).to_dict()
        
        self.num_user = self.rating_df.user_id.nunique()
        self.num_item = self.rating_df.product_id.nunique()
        self.max_user_degree = self.rating_df.user_degree.max()
        self.max_item_degree = self.rating_df.product_degree.max()
        print(f"***** Dataset Statistics *****")
        print("# of Users :", self.num_user)
        print("# of Items :", self.num_item, '\n')
        
        print(f"# of interactions : {self.rating_df.shape[0]}")
        print(f"# of Social Links : {self.trust_df.shape[0]}", '\n')
        
        print("Max User Degree :", self.max_user_degree)
        print("Max Item Degree :", self.max_item_degree)
        print(f"******************************", '\n')
        
        # Shuffle and split Rating dataframe (8:1:1 train/valid/test)
        self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, df_len=self.rating_df.shape[0], test=args.test_ratio, seed=args.seed, regen=args.regen)

        # Filter Social dataframe by Rating dataframe and Split
        # NOTE (A-3): these per-split social csvs are still written for inspection/tooling,
        # but they no longer feed the random walk -- see self.social_all below.
        self.social_train, self.rating_train = utils.generate_social_dataset(data_path, 'train', self.rating_train, self.trust_df, seed=args.seed, regen=args.regen)
        self.social_valid, self.rating_valid = utils.generate_social_dataset(data_path, 'valid', self.rating_valid, self.trust_df, seed=args.seed, regen=args.regen)
        self.social_test, self.rating_test = utils.generate_social_dataset(data_path, 'test', self.rating_test, self.trust_df, seed=args.seed, regen=args.regen)

        a = self.rating_train.groupby('user_id')['product_id'].apply(len).max()
        b = self.rating_valid.groupby('user_id')['product_id'].apply(len).max()
        c = self.rating_test.groupby('user_id')['product_id'].apply(len).max()
        self.min_item_len = c
        print(self.min_item_len)

        # Random Walk Sequence 생성
        # A-3: one shared social frame for every split (edges carry no labels). Walks start
        # from the split's target users but may traverse the whole graph; degree comes from
        # the global table so it stays consistent across splits.
        self.social_all = self.trust_df[['user_id_1', 'user_id_2']]
        self.random_walk_train = utils.generate_social_random_walk_sequence(data_path, self.rating_train, self.social_all, walk_length=args.user_seq_len, augs=args.augs, data_split_seed=args.seed, split='train', regen=args.regen, user_degree_dic=self.user_degree_dic)
        self.random_walk_valid = utils.generate_social_random_walk_sequence(data_path, self.rating_valid, self.social_all, walk_length=args.user_seq_len, augs=1, data_split_seed=args.seed, split='valid', regen=args.regen, user_degree_dic=self.user_degree_dic)
        self.random_walk_test = utils.generate_social_random_walk_sequence(data_path, self.rating_test, self.social_all, walk_length=args.user_seq_len, augs=1, data_split_seed=args.seed, split='test', regen=args.regen, user_degree_dic=self.user_degree_dic)

        # 모델 입력을 위한 최종 데이터셋 구성(rating)
        if args.regen=='train':
            train_regen=True
            valid_regen=False
            test_regen=False
        elif args.regen in ['all','rw','total']:
            train_regen=True
            valid_regen=True
            test_regen=True
        else:
            train_regen=False
            valid_regen=False
            test_regen=False
            
        # A-3: context_rating is ALWAYS the train split -> encoder never reads held-out labels.
        self.total_train, self.used_pairs = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_train, rating_split=self.rating_train, context_rating=self.rating_train, rating_matrix=self.rating_matrix, user_degree_dic=self.user_degree_dic, product_degree_dic=self.product_degree_dic,
                                                                               seed=args.seed, split='train', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user, min_item_len=self.min_item_len, regen=train_regen, neg=args.neg)

        # valid: same neg setting as test so the valid metric is comparable to the test metric.
        self.total_valid, _ = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_valid, rating_split=self.rating_valid, context_rating=self.rating_train, rating_matrix=self.rating_matrix, user_degree_dic=self.user_degree_dic, product_degree_dic=self.product_degree_dic,
                                                                 seed=args.seed, split='valid', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user, min_item_len=self.min_item_len, regen=valid_regen, neg=False, used_pairs=self.used_pairs, drop_cold_eval=getattr(args, 'drop_cold_eval', False))

        self.total_test, _ = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_test, rating_split=self.rating_test, context_rating=self.rating_train, rating_matrix=self.rating_matrix, user_degree_dic=self.user_degree_dic, product_degree_dic=self.product_degree_dic,
                                                                seed=args.seed, split='test', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user, min_item_len=self.min_item_len, regen=test_regen, neg=False, used_pairs=self.used_pairs, drop_cold_eval=getattr(args, 'drop_cold_eval', False))


def _get_args():
    import argparse
    p = argparse.ArgumentParser(description="SoFT data preparation (standalone runner)")
    p.add_argument('--dataset', type=str, required=True, help="ciao / epinions / ...")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--regen', type=str, default='no', help="[no, all, rw, total, train]")
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--user_seq_len', type=int, default=30, help="random walk sequence length")
    p.add_argument('--item_per_user', type=int, default=5)
    p.add_argument('--augs', type=int, default=1, help="train anchor augmentation factor")
    p.add_argument('--neg', action='store_true', help="add negative samples to the train target set")
    p.add_argument('--drop_cold_eval', action='store_true',
                   help="exclude anchors with 0 training interactions from valid/test")
    return p.parse_args()


if __name__ == '__main__':
    import random
    import numpy as np
    import torch
    args = _get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    DatasetMaking(args)
