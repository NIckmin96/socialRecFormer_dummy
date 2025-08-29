import os
import argparse
import pandas as pd
import numpy as np
from scipy.io import loadmat

def prepare_org_data(data, side_info=True):
    print("Processing _org.csv data...")
    # data_dir = os.path.join('dataset',data)
    data_dir = data
    data = data_dir.split('/')[-1]
    if data=='ciao_timestamp':
        rating_mat = loadmat(os.path.join(data_dir, 'rating_with_timestamp.mat'))
        trust_mat = loadmat(os.path.join(data_dir, 'trustnetwork.mat'))

        rating_arr = rating_mat['rating'].astype(np.int64)
        trust_arr = trust_mat['trustnetwork'].astype(np.int64)

        rating_df = pd.DataFrame(rating_arr, columns=['user_id','product_id','category_id','rating','helpfullness','timestamp'])
        trust_df = pd.DataFrame(trust_arr, columns=['user_id_1','user_id_2'])
        if side_info:
            rating_df = rating_df[['user_id','product_id','category_id','rating','timestamp']]
        else:
            rating_df = rating_df[['user_id','product_id','rating']]
        
    elif data=='ciao':
        rating_mat = loadmat(os.path.join(data_dir, 'rating.mat'))
        trust_mat = loadmat(os.path.join(data_dir, 'trustnetwork.mat'))

        rating_arr = rating_mat['rating'].astype(np.int64)
        trust_arr = trust_mat['trustnetwork'].astype(np.int64)

        rating_df = pd.DataFrame(rating_arr, columns=['user_id', 'product_id', 'category_id', 'rating', 'helpfulness'])
        trust_df = pd.DataFrame(trust_arr, columns=['user_id_1','user_id_2'])
        if side_info:
            rating_df = rating_df[['user_id','product_id','category_id','rating']]
        else:
            rating_df = rating_df[['user_id','product_id', 'rating']]
        # save dataframe to csv
        rating_df.to_csv(os.path.join(data_dir, 'rating_org.csv'), index=False)
        trust_df.to_csv(os.path.join(data_dir, 'trustnetwork_org.csv'), index=False)
        
    elif data=='epinions':
        rating_mat = loadmat(os.path.join(data_dir, 'rating_with_timestamp.mat'))
        trust_mat = loadmat(os.path.join(data_dir, 'trustnetwork.mat'))
        rating_arr = rating_mat['rating_with_timestamp'].astype(np.int64)
        trust_arr = trust_mat['trust'].astype(np.int64)

        rating_df = pd.DataFrame(rating_arr, columns=['user_id','product_id','category_id','rating','helpfullness','timestamp'])
        trust_df = pd.DataFrame(trust_arr, columns=['user_id_1','user_id_2'])
        if side_info:
            rating_df = rating_df[['user_id','product_id','category_id','rating','timestamp']]
        else:
            rating_df = rating_df[['user_id','product_id','rating']]
        
    elif data=='yelp':
        # 원본 데이터셋 새로 불러오기
        rating_chunk = pd.read_json(os.path.join(data_dir, 'yelp_academic_dataset_review.json'), lines=True, chunksize=50000)
        rating_df = pd.DataFrame()
        for chunk in rating_chunk:
            rating_df = pd.concat([rating_df, chunk], axis=0)

        social_chunk = pd.read_json(os.path.join(data_dir, 'yelp_academic_dataset_user.json'), lines=True, chunksize=50000)
        trust_df = pd.DataFrame()
        for chunk in social_chunk:
            trust_df = pd.concat([trust_df, chunk], axis=0)

        if side_info:
            business_chunk = pd.read_json(os.path.join(data_dir, 'yelp_academic_dataset_business.json'), lines=True, chunksize=50000)
            business = pd.DataFrame()
            for chunk in business_chunk:
                business = pd.concat([business, chunk], axis=0)    

        # rating
        print("'Rating' shape before processing :", rating_df.shape)
        rating_df = rating_df[['user_id','business_id','stars','date']]
        rating_df = rating_df.drop_duplicates(['user_id','business_id'])
        rating_df = rating_df.dropna(how='any')
        rating_df.columns = ['user_id','product_id','rating','timestamp']
        print("'Rating' shape after processing :", rating_df.shape)
        items = rating_df.product_id.unique()

        # social
        print("'Trust' shape before processing :", trust_df.shape)
        trust_df = trust_df[['user_id','friends']]
        trust_df['friends'] = trust_df['friends'].str.split(',')
        trust_df = trust_df.explode('friends')
        trust_df = trust_df.drop_duplicates()
        trust_df = trust_df.dropna(how='any')
        trust_df.columns = ['user_id_1','user_id_2']
        print("'Trust' shape after processing :", trust_df.shape)

        if side_info:
            # Filter 3 : Filter by 'rating df'
            business = business[business.business_id.isin(items)]
            # business(category)
            print("'Business' shape before processing :", business.shape)
            business = business[['business_id','categories']]
            business['categories'] = business['categories'].str.split(',')
            business = business.explode('categories')
            business = business.drop_duplicates()
            business = business.dropna(how='any')
            business.columns = ['product_id','category_id']
            print("'Business' shape after processing :", business.shape)
            # merge [rating, business]
            rating_df = pd.merge(rating_df, business, on='product_id', how='left')
        
    elif data=='douban': # small douban
        rating_df = pd.read_csv(os.path.join(data_dir, 'user_movie.dat'), sep='\t', header=None)
        rating_df.columns = ['user_id','product_id','rating']
        trust_df = pd.read_csv(os.path.join(data_dir, 'user_user.dat'), sep='\t', header=None)
        trust_df = trust_df.iloc[:,:2]
        trust_df.columns = ['user_id_1','user_id_2']
        
    elif data=='Douban':
        trust_dir = os.path.join(data_dir,'socialnet','socialnet.tsv')
        data_dir = os.path.join(data_dir, 'movie')
        rating_df = pd.read_csv(os.path.join(data_dir, 'douban_movie.tsv'), sep='\t')
        trust_df = pd.read_csv(trust_dir, sep='\t')
        rating_df.columns = ['user_id','product_id','rating','timestamp']
        trust_df.columns = ['user_id_1','user_id_2','Weight']
        trust_df = trust_df.drop(columns='Weight')
        
    # save dataframe to csv
    rating_df.to_csv(os.path.join(data_dir, 'rating_org.csv'), index=False)
    trust_df.to_csv(os.path.join(data_dir, 'trustnetwork_org.csv'), index=False)
    
    print("###### Data Source Processed ######")
    print(f"Num of users : {rating_df.user_id.nunique()}")
    print(f"Num of items : {rating_df.product_id.nunique()}")
    print("###################################\n")

    return rating_df, trust_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='ciao')
    parser.add_argument('--side', type=bool, default=True)
    args = parser.parse_args()
    data_dir = os.path.join('dataset',args.data)
    
    prepare_org_data(data_dir, args.side)
