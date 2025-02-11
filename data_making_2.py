import os
import argparse
import data_utils_2 as utils

class DatasetMaking:
    def __init__(self, dataset, seed, user_seq_len, item_seq_len, return_params, train_augs, test_augs, regenerate):
        # self.args = args
        data_path = os.getcwd() + '/dataset/' + dataset
        # create fundamental dataframe (Rating / Social)
        self.rating_df, self.trust_df = utils.mat_to_csv(data_path)
        
        # Shuffle and split Rating dataframe
        # self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, seed=seed)
        self.rating_train, self.rating_test = utils.shuffle_and_split_dataset(data_path, seed=seed, regenerate=regenerate)

        # Filter Social dataframe by Rating dataframe and Split
        self.social_train = utils.generate_social_dataset(data_path, 'train', self.rating_train, seed=seed, regenerate=regenerate)
        self.social_test = utils.generate_social_dataset(data_path, 'test', self.rating_test, seed=seed, regenerate=regenerate)
        # self.social_valid = utils.generate_social_dataset(data_path, 'valid', self.rating_valid, seed=seed)

        # User Degree Table
        self.user_degree_train = utils.generate_user_degree_table(data_path, self.social_train, split='train', seed=seed, regenerate=regenerate)
        self.user_degree_test = utils.generate_user_degree_table(data_path, self.social_test, split='test', seed=seed, regenerate=regenerate)
        # self.user_degree_valid = utils.generate_user_degree_table(data_path, self.social_valid, split='valid', seed=seed)

        # Item Degree Table
        self.item_degree_train = utils.generate_item_degree_table(data_path, self.rating_train, split='train', seed=seed, regenerate=regenerate)
        self.item_degree_test = utils.generate_item_degree_table(data_path, self.rating_test, split='test', seed=seed, regenerate=regenerate)
        # self.item_degree_valid = utils.generate_item_degree_table(data_path, self.rating_valid, split='valid', seed=seed)

        # user-item table(based on user)
        self.user_item_table_train = utils.generate_interacted_items_table(data_path, self.rating_train, self.item_degree_train, split='train', seed=seed, regenerate=regenerate)
        self.user_item_table_test = utils.generate_interacted_items_table(data_path, self.rating_test, self.item_degree_test, split='test', seed=seed, regenerate=regenerate)
        # self.user_item_table_valid = utils.generate_interacted_items_table(data_path, self.rating_valid, self.item_degree_valid, split='valid', seed=seed)

        ###################################################################### regenerate(Y/N) ######################################################################

        # Random Walk Sequence 생성
        ############################################################ [추가] Sequence Augmentation Argument : train_augs ###########################################################################
        ############################################################ [FIXME] [추가] Sequence Augmentation Argument : test_augs (train_augs에 비례하도록 설정) #########################################
        ############################################################ [추가] train dataset 생성 시, test dataset 재생성하지 않도록 수정 #########################################
        self.random_walk_train, rw_train_path = utils.generate_social_random_walk_sequence(data_path, self.social_train, self.user_degree_train, walk_length=user_seq_len, data_split_seed=seed, split='train', return_params=return_params, train_augs=train_augs, test_augs=False, regenerate=regenerate)
        self.random_walk_test, rw_test_path = utils.generate_social_random_walk_sequence(data_path, self.social_test, self.user_degree_test, walk_length=user_seq_len, data_split_seed=seed, split='test', return_params=return_params, train_augs=train_augs, test_augs=test_augs, regenerate=regenerate)
        # self.random_walk_valid = utils.generate_social_random_walk_sequence(data_path, self.social_valid, self.user_degree_valid, walk_length=user_seq_len, data_split_seed=seed, split='valid', return_params=return_params, train_augs=train_augs, test_augs=False, regenerate=regenerate)

        # Random Walk Sequence 중복 제거 + train/test에서 겹치는 경우 train에서 제거
        self.random_walk_train, self.random_walk_test = utils.remove_duplicated_social_random_walk_sequence(data_path, self.random_walk_train, self.random_walk_test, rw_train_path, rw_test_path)
        
        # 모델 입력을 위한 최종 데이터셋 구성
        self.total_test, test_user_item = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_test, item_df=self.user_item_table_test, seed=seed, split='test', random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, train_augs=train_augs, test_augs=test_augs, regenerate=regenerate)
        self.total_train = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_train, item_df=self.user_item_table_train, seed=seed, split='train', random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, train_augs=train_augs, test_augs=False, regenerate=regenerate, test_user_item=test_user_item)
        # self.total_valid = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_valid, item_df=self.user_item_table_valid, seed=seed, split='valid', random_walk_len=user_seq_len, item_seq_len=item_seq_len, return_params=return_params, train_augs=train_augs, test_augs=False, regenerate=regenerate)
        