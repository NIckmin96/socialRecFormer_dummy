import os
import argparse
import data_utils_2 as utils

class DatasetMaking:
    def __init__(self, args):
        # self.args = args
        data_path = os.getcwd() + '/dataset/' + args.dataset
        # create fundamental dataframe (Rating / Social)
        self.rating_df, self.trust_df = utils.mat_to_csv(data_path, args.regen)
        
        # Shuffle and split Rating dataframe
        self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, test=args.test_ratio, seed=args.seed, regen=args.regen)

        # Filter Social dataframe by Rating dataframe and Split
        self.social_train, self.rating_train = utils.generate_social_dataset(data_path, 'train', self.rating_train, self.trust_df,  seed=args.seed, regen=args.regen)
        self.social_valid, self.rating_valid = utils.generate_social_dataset(data_path, 'valid', self.rating_valid, self.trust_df, seed=args.seed, regen=args.regen)
        self.social_test, self.rating_test = utils.generate_social_dataset(data_path, 'test', self.rating_test, self.trust_df, seed=args.seed, regen=args.regen)
        self.num_user = max(self.rating_train.user_id.max(), self.rating_valid.user_id.max(), self.rating_test.user_id.max())
        self.num_item = max(self.rating_train.product_id.max(), self.rating_valid.product_id.max(), self.rating_test.product_id.max())

        # User Degree Table
        self.user_degree_train, max_user_degree_train = utils.generate_user_degree_table(data_path, self.social_train, split='train', seed=args.seed, regen=args.regen)
        self.user_degree_valid, max_user_degree_valid = utils.generate_user_degree_table(data_path, self.social_valid, split='valid', seed=args.seed, regen=args.regen)
        self.user_degree_test, max_user_degree_test = utils.generate_user_degree_table(data_path, self.social_test, split='test', seed=args.seed, regen=args.regen)
        self.max_user_degree = max(max_user_degree_train, max_user_degree_valid, max_user_degree_test)
        print("Max User Degree :", self.max_user_degree, '\n')

        # Item Degree Table
        self.item_degree_train, max_item_degree_train = utils.generate_item_degree_table(data_path, self.rating_train, split='train', seed=args.seed, regen=args.regen)
        self.item_degree_valid, max_item_degree_valid = utils.generate_item_degree_table(data_path, self.rating_valid, split='valid', seed=args.seed, regen=args.regen)
        self.item_degree_test, max_item_degree_test = utils.generate_item_degree_table(data_path, self.rating_test, split='test', seed=args.seed, regen=args.regen)
        self.max_item_degree = max(max_item_degree_train, max_item_degree_valid, max_item_degree_test)
        print("Max Item Degree :", self.max_item_degree, '\n')

        # user-item table(based on user)
        self.user_item_table_train = utils.generate_interacted_items_table(data_path, self.rating_train, self.item_degree_train, split='train', seed=args.seed, regen=args.regen)
        self.user_item_table_valid = utils.generate_interacted_items_table(data_path, self.rating_valid, self.item_degree_valid, split='valid', seed=args.seed, regen=args.regen)
        self.user_item_table_test = utils.generate_interacted_items_table(data_path, self.rating_test, self.item_degree_test, split='test', seed=args.seed, regen=args.regen)

        # Random Walk Sequence 생성
        self.random_walk_test, rw_test_path, used_set = utils.generate_social_random_walk_sequence(data_path, self.rating_test, self.social_test, self.user_degree_test, self.item_degree_test, walk_length=args.seq_len, data_split_seed=args.seed, split='test', regen=args.regen, used_set=None)
        self.random_walk_valid, rw_valid_path, used_set = utils.generate_social_random_walk_sequence(data_path, self.rating_valid, self.social_valid, self.user_degree_valid, self.item_degree_valid, walk_length=args.user_seq_len, data_split_seed=args.seed, split='valid', regen=args.regen, used_set=used_set)
        self.random_walk_train, rw_train_path, _ = utils.generate_social_random_walk_sequence(data_path, self.rating_train, self.social_train, self.user_degree_train, self.item_degree_train, walk_length=args.user_seq_len, data_split_seed=args.seed, split='train', regen=args.regen, used_set=used_set)

        # Random Walk Sequence 중복 제거 + train/test에서 겹치는 경우 train에서 제거
        self.random_walk_train, self.random_walk_valid, self.random_walk_test = utils.remove_duplicated_social_random_walk_sequence(self.random_walk_train, self.random_walk_valid, self.random_walk_test, rw_train_path, rw_valid_path, rw_test_path, args.regen)
        
        # 모델 입력을 위한 최종 데이터셋 구성(rating)
        self.total_test, self.test_user_item = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_test, item_df=self.user_item_table_test,
                                                                                    seed=args.seed, split='test', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user,
                                                                                    return_params=args.return_params, train_augs=args.train_augs, test_augs=args.test_augs, regen=args.regen)
        
        self.total_valid, self.valid_user_item = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_valid, item_df=self.user_item_table_valid,
                                                                                    seed=args.seed, split='valid', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user,
                                                                                    return_params=args.return_params, train_augs=args.train_augs, test_augs=args.test_augs, regen=args.regen)
        
        self.user_item_dic = utils.union_user_item_dict(self.test_user_item, self.valid_user_item)

        self.total_train = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_train, item_df=self.user_item_table_train,
                                                                seed=args.seed, split='train', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user,
                                                                 return_params=args.return_params, train_augs=args.train_augs, test_augs=args.test_augs, rating_thres=args.rating_thres,
                                                                  regen=args.regen, test_user_item=self.user_item_dic)
        
        