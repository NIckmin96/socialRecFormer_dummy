
ciao={  
         # dataset의 전체 길이 (기록 및 확인용) (train:val:test = 8:1:1)
         "dataset":{
            "train":53393,
            "dev":8489,
            "test":8428,
         },
         "model":{
            "num_user": 7317,
            "num_item": 104975,
            "max_user_degree": 804,
            "max_item_degree": 721,
            "max_spd_value": 15,
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim 
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 3,
            "num_layers_dec": 5,
            "n_experts": 8,
            "topk": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.01,
            "warmup":40, 
            "lr_decay":"linear",
            "weight_decay":1e-1,
            "num_epochs":100,
            "patience":10, 
            "alpha":1,
            "beta":1,
            "gamma":1,
            "baseline_rmse":0.974,
            "baseline_mae":0.7323
         },
     }

ciao_timestamp={  
         # dataset의 전체 길이 (기록 및 확인용) (train:val:test = 8:1:1)
         "dataset":{
            "train":53393,
            "dev":8489,
            "test":8428,
         },
         "model":{
            "num_user": 2342,
            "num_item": 77524,
            "max_user_degree": 453,
            "max_item_degree": 334,
            "max_spd_value": 15,
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim 
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 3,
            "num_layers_dec": 5,
            "n_experts": 8,
            "topk": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.01,
            "warmup":40, 
            "lr_decay":"linear",
            "weight_decay":1e-1,
            "num_epochs":100,
            "patience":10, 
            "alpha":1,
            "beta":1,
            "gamma":1,
            "baseline_rmse":0.974,
            "baseline_mae":0.7323
         },
     }

epinions={
         "dataset":{
             "train":560000,
             "dev":38000,
             "test":38000,
         },
         "model":{
            "num_user": 18097,
            "num_item": 261679,
            "max_user_degree": 2026,
            "max_item_degree": 1155,
            "max_spd_value": 15,
            "d_model": 56,          # MHA dim (Linear modules in Attention Network) & Embedding dim num_workers
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 3,
            "num_layers_dec": 5,
            "n_experts": 8,
            "topk": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.01,
            "warmup":80, 
            "lr_decay":"cos",
            "weight_decay":1e-1,
            "eval_frequency":400, 
            "num_epochs":100,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
            "patience":10, 
            "alpha":1,
            "beta":3,
            "gamma":3,
            "baseline_mae":0.8383,
            "baseline_rmse":1.0972
         },
     }

yelp={
         "model":{
            "num_user": 75710,
            "num_item": 114854,
            "max_user_degree": 4118,
            "max_item_degree": 717,
            "max_spd_value": 15,
            "d_model": 32,          # MHA dim (Linear modules in Attention Network) & Embedding dim num_workers
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 3,
            "num_layers_dec": 5,
            "n_experts": 8,
            "topk": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.01,
            "warmup":80, 
            "lr_decay":"cos",
            "weight_decay":1e-1,
            "eval_frequency":400, 
            "num_epochs":100,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
            "patience":10, 
            "alpha":1,
            "beta":3,
            "gamma":3,
            "baseline_mae":0.8383,
            "baseline_rmse":1.0972
         },
     }

douban={
         "model":{
            "num_user": 75710,
            "num_item": 114854,
            "max_user_degree": 4118,
            "max_item_degree": 717,
            "max_spd_value": 15,
            "d_model": 48,          # MHA dim (Linear modules in Attention Network) & Embedding dim num_workers
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 3,
            "num_layers_dec": 5,
            "n_experts": 8,
            "topk": 1
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.01,
            "warmup":80, 
            "lr_decay":"cos",
            "weight_decay":1e-1,
            "eval_frequency":400, 
            "num_epochs":100,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
            "patience":10, 
            "alpha":1,
            "beta":3,
            "gamma":3,
            "baseline_mae":0.8383,
            "baseline_rmse":1.0972
         },
     },

Douban={
         "model":{
            "num_user": 72000,
            "num_item": 80269,
            "max_user_degree": 2982,
            "max_item_degree": 21316,
            "max_spd_value": 15,
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim num_workers
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 3,
            "num_layers_dec": 5,
            "n_experts": 8,
            "topk": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.01,
            "warmup":80, 
            "lr_decay":"cos",
            "weight_decay":1e-1,
            "eval_frequency":400, 
            "num_epochs":100,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
            "patience":10, 
            "alpha":1,
            "beta":3,
            "gamma":3,
            "baseline_mae":0.8383,
            "baseline_rmse":1.0972
         },
     }



Config = {
    "ciao":ciao,
    "ciao_timestamp":ciao_timestamp,
    "epinions":epinions,
    "yelp":yelp,
    "douban":douban,
    "Douban":Douban
}