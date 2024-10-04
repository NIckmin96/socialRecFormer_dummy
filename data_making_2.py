import os
import argparse
import data_utils_2 as utils

# def get_args():
#     parser = argparse.ArgumentParser(description='Data preparation(preprocess) for Transformer input')
#     parser.add_argument("--dataset", type=str, default="epinions", help="ciao // epinions")
#     parser.add_argument("--seed", type=int, default=42, help="random seed, used in dataset split")
#     parser.add_argument("--test_ratio", type=float, default=0.1, help="percentage of valid/test dataset")
#     parser.add_argument("--user_seq_len", type=int, default=30, help="random walk seqeunce length (encoder's input length)")
#     parser.add_argument("--item_seq_len", type=int, default=100, help="item list length (decoder's input length)")
#     parser.add_argument("--return_params", type=int, default=1)
#     parser.add_argument("--train_augs", type=int, default=10)

#     args = parser.parse_args()

#     return args

# def main():
#     args = get_args()
#     data_path = os.getcwd() + '/dataset/' + args.dataset
#     data_making = DatasetMaking(data_path)

#     return data_making

class DatasetMaking:
    def __init__(self, dataset, seed, user_seq_len, item_seq_len, return_params, train_augs, regenerate):
        # self.args = args
        data_path = os.getcwd() + '/dataset/' + dataset
        # create fundamental dataframe (Rating / Social)
        self.rating_df, self.trust_df = utils.mat_to_csv(data_path)
        # Shuffle and split Rating dataframe
        self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, seed=seed)
        # Filter Social dataframe by Rating dataframe and Split
        self.social_train = utils.generate_social_dataset(data_path, 'train', self.rating_train, seed=seed)
        self.social_valid = utils.generate_social_dataset(data_path, 'valid', self.rating_valid, seed=seed)
        self.social_test = utils.generate_social_dataset(data_path, 'test', self.rating_test, seed=seed)
        # User Degree Table
        self.user_degree_train = utils.generate_user_degree_table(data_path, self.social_train, split='train', seed=seed)
        self.user_degree_valid = utils.generate_user_degree_table(data_path, self.social_valid, split='valid', seed=seed)
        self.user_degree_test = utils.generate_user_degree_table(data_path, self.social_test, split='test', seed=seed)
        # Item Degree Table
        self.item_degree_train = utils.generate_item_degree_table(data_path, self.rating_train, split='train', seed=seed)
        self.item_degree_valid = utils.generate_item_degree_table(data_path, self.rating_valid, split='valid', seed=seed)
        self.item_degree_test = utils.generate_item_degree_table(data_path, self.rating_test, split='test', seed=seed)
        # user-item table(based on user)
        self.user_item_table_train = utils.generate_interacted_items_table(data_path, self.rating_train, self.item_degree_train, split='train', seed=seed)
        self.user_item_table_valid = utils.generate_interacted_items_table(data_path, self.rating_valid, self.item_degree_valid, split='valid', seed=seed)
        self.user_item_table_test = utils.generate_interacted_items_table(data_path, self.rating_test, self.item_degree_test, split='test', seed=seed)

        # Random Walk Sequence 생성
        ############### Sequence Augmentation Argument 추가 : train_times ###############
        self.random_walk_train, self.random_walk_valid, self.random_walk_test = utils.generate_social_random_walk_sequence(data_path, [self.social_train, self.social_valid, self.social_test],
                                                                                                                            self.user_degree_train, walk_length=user_seq_len, data_split_seed=seed, split='train', return_params=return_params, train_augs=train_augs, regenerate=regenerate)
        # self.random_walk_train = utils.generate_social_random_walk_sequence(data_path, self.social_train, self.user_degree_train, walk_length=user_seq_len, data_split_seed=seed, split='train', return_params=return_params, train_augs=train_augs, regenerate=regenerate)
        # self.random_walk_valid = utils.generate_social_random_walk_sequence(data_path, self.social_valid, self.user_degree_valid, walk_length=user_seq_len, data_split_seed=seed, split='valid', return_params=return_params, regenerate=regenerate)
        # self.random_walk_test = utils.generate_social_random_walk_sequence(data_path, self.social_test, self.user_degree_test, walk_length=user_seq_len, data_split_seed=seed, split='test', return_params=return_params, regenerate=regenerate)
        ############# 모델 입력을 위한 최종 데이터셋 구성
        user_df = {'user_train':self.random_walk_train, 'user_valid':self.random_walk_valid, 'user_test':self.random_walk_test}
        item_df = {'item_train':self.user_item_table_train, 'item_valid':self.user_item_table_valid, 'item_test':self.user_item_table_test}
        self.total_train, self.total_valid, self.total_test = utils.generate_input_sequence_data(data_path=data_path, user_df = user_df, item_df = item_df, seed=seed, random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, train_augs=train_augs, regenerate=regenerate)
        # self.total_train = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_train, item_df=self.user_item_table_train, seed=seed, split='train', random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, train_augs=train_augs, regenerate=regenerate)
        # self.total_valid = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_valid, item_df=self.user_item_table_valid, seed=seed, split='valid', random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, regenerate=regenerate)
        # self.total_test = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_test, item_df=self.user_item_table_test, seed=seed, split='test', random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, regenerate=regenerate)