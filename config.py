
ciao={  
         # dataset의 전체 길이 (기록 및 확인용) (train:val:test = 8:1:1)
         "dataset":{
            "train":53393,
            "dev":8489,
            "test":8428,
         },
         "model":{
            "num_user": 7317,
            "max_degree_user": 804,
            "num_item": 105114,
            "max_degree_item": 915,
            "max_spd_value": 15,
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim 
            "d_ffn": 256,            # FFN dim
            "num_heads": 4,
            "dropout": 0.1,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 4,
            "num_layers_dec": 4
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.0001,
            "warmup":40, 
            "lr_decay":"linear",
            "weight_decay":0,
            "num_epochs":100,
            "patience":10, 
            "alpha":1,
            "beta":3,
            "gamma":3,
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
            "num_user": 18098,
            "max_degree_user": 2026,
            "num_item": 261679,
            "max_degree_item": 1440,
            "max_spd_value": 15,
            "d_model": 64,          # MHA dim (Linear modules in Attention Network) & Embedding dim num_workers
            "d_ffn": 256,            # FFN dim
            "num_heads": 2,
            "dropout": 0.2,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
            "num_layers_enc": 2,
            "num_layers_dec": 2
         },
         "training":{
            "batch_size":128,        # total_train_step: 835 (1 epoch 당 `len(train_dataset) / batch_size`)
            "optimizer":"adamw",
            "learning_rate":0.0001,
            "warmup":80, 
            "lr_decay":"linear",
            "weight_decay":0,
            "eval_frequency":400, 
            "num_epochs":100,
            "num_eval_steps":849,   # total_valid_sample / total_epoch
            "patience":10, 
            "alpha":1,
            "beta":3,
            "gamma":3,
            "baseline_rmse":0.8383,
            "baseline_mae":1.0972
         },
     }

Config = {
    "ciao":ciao,
    "epinions":epinions,
}