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
        