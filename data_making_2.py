import os
import scipy.sparse as sparse
import data_utils_2 as utils

class DatasetMaking:
    def __init__(self, args):
        # self.args = args
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
        
        # Shuffle and split Rating dataframe
        self.rating_train, self.rating_valid, self.rating_test = utils.shuffle_and_split_dataset(data_path, df_len=self.rating_df.shape[0], test=args.test_ratio, seed=args.seed, regen=args.regen)
        self.max_item_len = max(self.rating_train.groupby('user_id')['product_id'].apply(len).max(), self.rating_valid.groupby('user_id')['product_id'].apply(len).max(), self.rating_test.groupby('user_id')['product_id'].apply(len).max())
        print(self.max_item_len)

        # Filter Social dataframe by Rating dataframe and Split
        self.social_train, self.rating_train = utils.generate_social_dataset(data_path, 'train', self.rating_train, self.trust_df, seed=args.seed, regen=args.regen)
        self.social_valid, self.rating_valid = utils.generate_social_dataset(data_path, 'valid', self.rating_valid, self.trust_df, seed=args.seed, regen=args.regen)
        self.social_test, self.rating_test = utils.generate_social_dataset(data_path, 'test', self.rating_test, self.trust_df, seed=args.seed, regen=args.regen)

        # Random Walk Sequence 생성
        self.random_walk_train, rw_train_path = utils.generate_social_random_walk_sequence(data_path, self.rating_train, self.social_train, walk_length=args.user_seq_len, augs=args.augs, data_split_seed=args.seed, split='train', regen=args.regen)
        self.random_walk_valid, rw_valid_path = utils.generate_social_random_walk_sequence(data_path, self.rating_valid, self.social_valid, walk_length=args.user_seq_len, augs=1, data_split_seed=args.seed, split='valid', regen=args.regen)
        self.random_walk_test, rw_test_path = utils.generate_social_random_walk_sequence(data_path, self.rating_test, self.social_test, walk_length=args.user_seq_len, augs=1, data_split_seed=args.seed, split='test', regen=args.regen)
        
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
            
        self.total_train, self.total_train2, self.used_pairs = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_train, rating_split=self.rating_train, rating_matrix=self.rating_matrix, user_degree_dic=self.user_degree_dic, product_degree_dic=self.product_degree_dic,
                                                                               seed=args.seed, split='train', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user, max_item_len=self.max_item_len, regen=train_regen, neg=args.neg)
        
        self.total_valid, self.total_valid2, _ = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_valid, rating_split=self.rating_valid, rating_matrix=self.rating_matrix, user_degree_dic=self.user_degree_dic, product_degree_dic=self.product_degree_dic,
                                                                 seed=args.seed, split='valid', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user, max_item_len=self.max_item_len, regen=valid_regen, neg=True, used_pairs=self.used_pairs)
        
        self.total_test, self.total_test2, _ = utils.generate_input_sequence_data(data_path=data_path, rw_df=self.random_walk_test, rating_split=self.rating_test, rating_matrix=self.rating_matrix, user_degree_dic=self.user_degree_dic, product_degree_dic=self.product_degree_dic,
                                                                seed=args.seed, split='test', random_walk_len=args.user_seq_len, item_per_user=args.item_per_user, max_item_len=self.max_item_len, regen=test_regen, neg=True, used_pairs=self.used_pairs)
        
        

        
        
        