# requirements
pip==24.0
setuptools=59.5.0

# socialRecFormer_dummy
Reference : https://github.com/Bokyeong1001/SocialRecFormer

# Data Creation Process
run 'data_making.py' with proper arguments

## `data_making.py`
1. mat_to_csv
    1. Load 'trust_df'(social interaction data) & 'rating_df'(user item interaction data)
    2. reset_and_filter_data
        - Select common users between 'trust_df' & 'rating_df' and filter
        - Rearrange user id starting from '1'
    3. Create user-item rating matrix(/w `spares.lil_matrix`)
    4. Save processed data(`trust_df, rating_df, rating_matrix`)

2. shuffle_and_split_dataset
    - Shuffle `rating_df` into *train/valid/test* datatset and save

3. generate_social_dataset
    - Split social data(`trust_df`) by common users in rating data and save each(train/test/valid) social data
    
4. generate_user_degree_table
    1. Load social data(`trust_df`) and make graph data with it(`social_graph`)
    2. Get degrees of each user from `social_graph`
    3. save user_id to degree data(`degree_table`)

5. generate_item_degree_table
    1. Load rating data(`rating_df`) and group by item(product_id) and get degree and save data(`degree_table_item`)

6. generate_interacted_items_table
    - Generate a dataframe consisted of [user, item list, rating, item degree]

    |user id|item list|rating|item degree|  
    |---|---|---|---|
    |1|[3,4,5]|[3,3,4]|[1,2,3]|

    - Add zero padding row

    |user id|item list|rating|item degree|  
    |---|---|---|---|
    |0|[0,0,0]|[0,0,0]|[0,0,0]|

7. generate_interacted_user_table
    - Generate a dataframe consisted of [item, user list, rating]

    |item id|user list|rating|
    |---|---|---|
    |1|[3,4,5]|[3,3,4]|

8. generate_social_random_walk_sequence
    - Generate random user sequence starting from anchor user

    |anchor user|sequence|
    |---|---|
    |1|[3,4,5]|

    1. From all nodes of social graph, randomly select an `anchor(starting) node`
        - If train dataset, create 10 rows for each anchor node
        - Else only one row
    2. Sequentially pick random user(`find_next_node`)
        - Pick neighbor node of present selected node by uniform distribution
            - Previously selected node has specified probability(`RETURN_PARMS`)
            - But, not applied to all previous nodes only the node right before present node
        - Repeat this until user sequence length meets limit(`walk_length`)

9. generate_input_sequence_data
    - Create final data which will be used in model training(`total_df`)

    |anchor user|user sequence|user degree|item sequence|item degree|rating|user distance|
    |---|---|---|---|---|---|---|
    |1|[3,4,5]|[1,1,2]|[1,2,3]|[4,4,5]|[3,3,3]|[1,2,3]|

    1. Get item sequence for each user in user sequence and append items in a list
        - `item_list`
    2. Get item degrees for each user in user sequence and append degrees in a list
        - `degree_list`
    3. Drop duplicates in `item_list`
        - `item_list_removed_duplicate`
    4. Map each item in `item_list` to degree(`degree_list`)
        - `mapping_dict` ->  (item : degree)
        - `mapping_dict`'s values -> `degree_list_removed_duplicate`
    5. Map each item in `item_list` to related user(`user_item_list`)
        - `user_mapping_dict` -> (item:user)
    6. `slice_and_pad_list`
        - pad each list with 0 and make its length n-times of `item_seq_len` and slice
            - ex: len(item_list) = 110 & `item_seq_len`=30 $\rightarrow$ len(item_list)+[0]*10
            - `item_list_removed_duplicate`
            - `degree_list_removed_duplicate`
    7. Create `spd_matrix`(shortest distance between users)
        - Get a symmetric subset of total spd matrix(`spd_table`) shaped n_user_sequence $\times$ n_user_sequence
        
    8. Create rating matrix of 'current users' and 'sliced items'(`small_rating_matrix`)
    
    9. Sort them in a dataframe(`total_df`) and save
