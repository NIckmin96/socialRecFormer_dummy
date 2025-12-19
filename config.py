ciao_timestamp={  
         "model":{
            "enc_blocks":1, # fix
            "dec_blocks":2,
            "n_experts":4, # 마지막 튜닝
            "topk":3, # 마지막 튜닝
            "num_heads":6,
            "d_model":64*6,
            "d_ffn":256,
            "dropout": 0.2,
            "moe":True
         },
         "training":{
            "weight_decay_enc":9e-2,
            "weight_decay_dec":3e-2,
            "lr_enc":3e-2,
            "lr":4e-3,
            "num_epochs":300,
            "bs_enc":32, # fix
            "bs_dec":32
         },
     }

epinions={
         "model":{
            "enc_blocks":1, # 고정
            "dec_blocks":1, # 고정
            "n_experts":6, # 마지막 튜닝
            "topk":2, # 마지막 튜닝
            "num_heads":6, # 고정
            "d_model":32*6, # 고정
            "d_ffn":256, # 256이나 400근처
            "dropout": 0.1, # 고정
            "moe":True
         },
         "training":{
            "weight_decay_enc":5e-2, # 고정
            "weight_decay_dec":3e-2, # [1e-2 ~ 1e-1]
            "lr_enc":5e-3, # 고정
            "lr":1e-3, # 1e-3 근처
            "num_epochs":300,
            "bs_enc":32, # 고정
            "bs_dec":128 # [32,64,128,256]
         },
     }

yelp={
         "model":{
            "enc_blocks":1,
            "dec_blocks":1,
            "n_experts":2,
            "topk":1,
            "num_heads":4,
            "d_model":32*4,
            "d_ffn":256,
            "dropout": 0.2,
            "moe":True
         },
         "training":{
            "weight_decay_enc":4e-2,
            "weight_decay_dec":5e-3,
            "lr_enc":1e-2,
            "lr":1e-4,
            "num_epochs":300,
            "bs_enc":256,
            "bs_dec":256
         },
     }

Douban={
        "model":{
            "enc_blocks":1,
            "dec_blocks":1,
            "num_heads":6,
            "n_experts":6,
            "topk":2,
            "d_model":32*6,
            "d_ffn":256,
            "dropout": 0.3,
            "moe":True
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-3,
            "lr_enc":1e-2,
            "lr":5e-3,
            "num_epochs":300,
            "bs_enc":256,
            "bs_dec":256
         },
     }



Config = {
    "ciao_timestamp":ciao_timestamp,
    "epinions":epinions,
    "yelp":yelp,
    "Douban":Douban
}