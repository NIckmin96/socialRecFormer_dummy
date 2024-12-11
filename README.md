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

########################################################## Test-set Augmentation ##########################################################
# Augment Test set Data in proportion to train set data
- if `train_augs = 2`, then `test_augs = 2`
    - `data_utils_2` -> `generate_social_random_walk_sequence`

### `generate_social_random_walk_sequence` 수정
- train/valid/test 간 중복되는 sequence는 없음(처음부터 사용되는 데이터에서 anchor노드를 split하기 때문에 ->  `generate_social_dataset`에서 split 진행)
- 하지만, 동일한 데이터셋 내에서 중복되는 sequence가 생길수도 있음
    - Anchor node를 기준으로 random하게 sequence를 생성하지만, 중복을 확인하는 장치는 없음
    - 중복되는 sequence가 학습에 사용되면 overfitting의 위험성이 있음
    - 중복되는 경우가 현재는 확인된 바는 없지만, 장치를 만들어줄 필요 있음

# Model Train/Valid/Test Process
1. `main.py`
    1. `main` function
        1. Get Arguments + set hyper-parameters
        2. set log directory + set model checkpoint
        3. dataset load or creation + dataloader
        4. training preparation
            - optimizer(learning rate, weigt decay), hyperparameters(num epochs, train steps)
            - learning rate scheduler(OneCycleLR)
            - Tensorboard Writer
    2. `train` function


# Model Architecture

## Transformer

### Encoder
- Social Representation(**User-User**)

1. Encoder Layers(`blocks - encoder_layer.py - EncoderLayer`)
    1. Create User-Input Embedding Vectors(`layers - encoding_modules.py - SocialNodeEnocder`)
        - User-Input Embedding = User Embedding + User-Degree Embedding

    2. Create Attention Score Mask(`model_utils.py - generate_attn_pad_mask`)
        - input : batched user sequence
        - user sequence에서 0이 아닌 부분에 대해 마스크 생성
            - 추후, attention score계산시에 mask에 해당하지 않는 부분(`node==0`)에 대해 마스킹 적용
            - [batch size x sequence length x sequence length]

    3. Create Attention Bias(`layers - encoding_modules.py - SpatialEncoder`)
        - batched spd_table(shortest path)를 변형해 attn_bias(attention bias)생성
        - spd_table을 multi head attention의 head수에 맞게 확장 시킨 것
            - spd_table : [batch size x len_user_sequence x len_item_sequence]
            - attn_bias : [batch size x num_heads x len_user_sequence x len_item_sequence]

    4. Layer Normalization

    3. Multi Head Attention(`layers - multi_head_attention.py - MultiHeadAttention`)
        1. Create Q,K,V tensors projecting User-Input Embedding to different linear layer
            - [batch_size x seq_length x d_model] $\rightarrow$ [batch_size x seq_length x d_model]

        2. Split Q,K,V tensors by number of heads
            - [batch_size x num_heads x seq_length x d_tensor(= d_model/num_heads)]

        3. Reshape mask to correspond with 'Attention Score' Apply mask for multi-head attention
            - mask : [batch_size x len_q x len_k] -> [batch_size x num_heads x len_q x len_k]

        4. Scaled Dot product Attention(`layers - multi_head_attnetion.py - ScaledDotProductAttention`)
            - (attention) score = $\frac{Q\cdot K^{T}}{\sqrt{\text{d-tensor}}}$

            - attn_bias (attention bias)
                - attn_bias(shortest path를 head수에 맞게 확장)
                - attn_bias가 0인 경우에는 1, 그렇지 않은 경우에는 attn_bias 제곱의 역수 (1/attn_bias**2)

            - loss
                - 계산된 (attention) score와 attn_bias의 mse loss

            - attention output
                - (attention) score를 softmax함수에 태워, 가중치를 계산 후 V matrix와 연산 후, concat
        
        5. 1~4번까지의 과정을 거쳐 계산된 [attention output, spd_loss]값 return

    4. Dropout + residual connection

    5. Layer Normalization

    6. Feed Forward Network(`layers - feed_forward_network.py - FeedForwardNetwork`)
        - Linear Combination Layer
            - d_model $\rightarrow$ d_ffn

        - Activation Function 1
            - `GeLU`

        - Dropout

        - Activation Function 2
            - `ReLU`

    7. Dropout + residual connection

2. Loop Encoder Layers and append `spd_loss` per loop

3. Return `encoder output`, `mean value of spd_loss`

### Decoder
- Item Representation(**User-Item**)

1. Decoder Layers(loop by 'num_layers')

    1. Create Item Embedding Vectors
        - Item node embedding + item degree embedding

    2. Create self-attention score mask
        - input : batched item sequence
        - **item sequence**에서 0에 해당되지 않는 부분에 대한 mask 생성
            - 추후, attention score계산시에 mask에 해당하지 않는 부분(`node==0`)에 대해 마스킹 적용
            - [batch size x item sequence x item sequence]

    3. Create cross-attention score mask
        - input  : batched 'item sequence' & batched 'user sequence'
        - **user sequence**에서 0에 해당되지 않는 부분에 대한 mask 생성
            - 추후, attention score계산시에 mask에 해당하지 않는 부분(`node==0`)에 대해 마스킹 적용
            - [batch size x item sequence x user sequence]
            - <U>[???] item sequence에 대한 masking이 안들어가는데 괜찮은 것인지</U>

    4. Create Decoder Attention bias
        - **Rating**
        - exchange **explicit item rating** into **implicit item rating** (0/1)
        - expand attention bias into multi-head attention output size
            - input : [batch size x user sequence x item sequence]
            - output : [batch size x num heads x user sequence x item sequence]

    5. Decoder Layers(Repeat by 'num_layers')
        1. Layer Norm
        
        2. Self Attention
            - Q,K,V from **Item Embedding**(Decoder)
            - Same Process with Encoder MHA afterwards

        3. Dropout + Residual Connection

        4. Layer Norm

        5. Cross Attention
            - **Q = Decoder self attention output**
            - **K,V = Encoder output**

        6. Dropout + Residual Connection

        ----------------------------
        - if **last layer(Prediction Layer)**:
            - Perform last Cross Attention
            - Return Attention output & loss

        - else:
            1. Layer Norm
            2. Feed Forward Network
            3. Residual Connection

2. Return Prediction Output & total mean loss
    - 각 decoder block에서 return되는 loss(decoder loss)
        1. 정답 rating값을 implicit화(0,1) -> (-1,1)
        2. Attention output과의 MAE 계산
        - decoder loss = MAE

## Transformer output
- Decoder output
    - [batch size x user sequence x item sequence]
- Encoder Loss
- Decoder Loss


## Model Optimization loss
1. `org_loss`
    - Calculte mean squared error between [Predicted Rating, Real Rating]
    - **Item rating==0**인 것들은 고려하지 않음

2. `new_loss`
    - mini batch의 첫번째 user에 해당하는 rating에 대해서만 MSE 계산
    - **Item rating==0**인 것들은 고려하지 않음

3. `loss`
    - `org_loss` + `new_loss` + 'alpha' + decoder loss * 'gamma' + encoder loss * 'beta'

## Validation / Test
- `new_loss` + encoder loss + decoder loss만을 가지고 validation/test 진행
- batch별 첫번째 sequence만을 가지고 validation/test