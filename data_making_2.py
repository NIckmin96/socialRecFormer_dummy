import os
import data_utils_2 as utils

class DatasetMaking:
    def __init__(self, args):
        # self.args = args
        data_path = os.getcwd() + '/dataset/' + args.dataset
        # create fundamental dataframe (Rating / Social)
        self.rating_df, self.trust_df = utils.mat_to_csv(data_path, args.regenerate)
        
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
        
        # Shuffle and split Rating dataframe
        self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, test=args.test_ratio, seed=args.seed, regenerate=args.regenerate)

        # Filter Social dataframe by Rating dataframe and Split
        self.social_train, self.rating_train = utils.generate_social_dataset(data_path, 'train', self.rating_train, self.trust_df,  seed=args.seed, regenerate=args.regenerate)
        self.social_valid, self.rating_valid = utils.generate_social_dataset(data_path, 'valid', self.rating_valid, self.trust_df, seed=args.seed, regenerate=args.regenerate)
        self.social_test, self.rating_test = utils.generate_social_dataset(data_path, 'test', self.rating_test, self.trust_df, seed=args.seed, regenerate=args.regenerate)

        # Random Walk Sequence 생성
        self.random_walk_train, rw_train_path = utils.generate_social_random_walk_sequence(data_path, self.rating_train, self.social_train, walk_length=args.user_seq_len, data_split_seed=args.seed, split='train', regenerate=args.regenerate)
        self.random_walk_valid, rw_valid_path = utils.generate_social_random_walk_sequence(data_path, self.rating_valid, self.social_valid, walk_length=args.user_seq_len, data_split_seed=args.seed, split='valid', regenerate=args.regenerate)
        self.random_walk_test, rw_test_path = utils.generate_social_random_walk_sequence(data_path, self.rating_test, self.social_test, walk_length=args.user_seq_len, data_split_seed=args.seed, split='test', regenerate=args.regenerate)

        # # Random Walk Sequence 중복 제거 + train/test에서 겹치는 경우 train에서 제거
        # self.random_walk_train, self.random_walk_valid, self.random_walk_test = utils.remove_duplicated_social_random_walk_sequence(self.random_walk_train, self.random_walk_valid, self.random_walk_test, rw_train_path, rw_valid_path, rw_test_path, args.regenerate)
        
        # 모델 입력을 위한 최종 데이터셋 구성(rating)
        self.total_test, self.used_pairs = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_test, rating_df=self.rating_df, seed=args.seed,
                                                                              split='test', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user,  regenerate=args.regenerate)
        
        self.total_valid, self.used_pairs = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_valid, rating_df=self.rating_df, seed=args.seed,
                                                                              split='valid', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user,  regenerate=args.regenerate, used_pairs=self.used_pairs)

        self.total_train, self.used_pairs = utils.generate_input_sequence_data(data_path=data_path, user_df=self.random_walk_train, rating_df=self.rating_df, seed=args.seed,
                                                                              split='train', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user,  regenerate=args.regenerate, used_pairs=self.used_pairs)
        
        