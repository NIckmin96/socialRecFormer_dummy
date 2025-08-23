import os
import argparse
import data_utils_2 as utils

class DatasetMaking:
    def __init__(self, args):
        # self.args = args
        data_path = os.getcwd() + '/dataset/' + args.dataset
        # create fundamental dataframe (Rating / Social)
        self.rating_df, self.trust_df = utils.mat_to_csv(data_path, args.regen)
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
        print("Max Item Degree :", self.max_item_degree, '\n')
        
        # Shuffle and split Rating dataframe
        self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, test=args.test_ratio, seed=args.seed, regen=args.regen)

        # Random Walk Sequence 생성
        self.random_walk_test, rw_test_path, used_set = utils.generate_rw_sequence(data_path, self.rating_test, walk_length=args.seq_len, data_split_seed=args.seed, split='test', regen=args.regen, used_set=set())
        self.random_walk_valid, rw_valid_path, used_set = utils.generate_rw_sequence(data_path, self.rating_valid, walk_length=args.seq_len, data_split_seed=args.seed, split='valid', regen=args.regen, used_set=used_set)
        self.random_walk_train, rw_train_path, _ = utils.generate_rw_sequence(data_path, self.rating_train, walk_length=args.seq_len, data_split_seed=args.seed, split='train', regen=args.regen, used_set=used_set)

        # Random Walk Sequence 중복 제거 + train/test에서 겹치는 경우 train에서 제거
        self.random_walk_train, self.random_walk_valid, self.random_walk_test = utils.remove_duplicated_random_walk_sequence(self.random_walk_train, self.random_walk_valid, self.random_walk_test, rw_train_path, rw_valid_path, rw_test_path, args.regen)
        
        # 모델 입력을 위한 최종 데이터셋 구성(rating)
        self.total_test = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_test, rating_df=self.rating_df, seed=args.seed, split='test', random_walk_len=args.seq_len, regen=args.regen)
        self.total_valid = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_valid, rating_df=self.rating_df, seed=args.seed, split='valid', random_walk_len=args.seq_len, regen=args.regen)
        self.total_train = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_train, rating_df=self.rating_df, seed=args.seed, split='train', random_walk_len=args.seq_len, regen=args.regen)
        
        