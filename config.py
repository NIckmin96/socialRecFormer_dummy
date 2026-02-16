ciao_timestamp={  
         "model":{
            "enc_blocks":1, # fix
            "dec_blocks":3, # fix
            "n_experts":5, # fix
            "topk":2, # fix
            "num_heads":8,
            "d_model":64*8,
            "d_ffn":512, # fix
            "dropout": 0.3, # fix
            "moe":True
         },
         "training":{
            "weight_decay_enc":6e-2, # fix
            "weight_decay_dec":3e-2, # fix
            "lr_enc":2e-3, # fix
            "lr":9e-4, # fix
            "num_epochs":300,
            "bs_enc":32, # fix
            "bs_dec":32 # fix
         },
     }

epinions={
         "model":{
            "enc_blocks":1, # 고정
            "dec_blocks":3, # 고정
            "n_experts":5, # fix
            "topk":2, # fix
            "num_heads":8, # 고정
            "d_model":64*8, # 고정
            "d_ffn":512, # 고정
            "dropout": 0.3, # fix
            "moe":True
         },
         "training":{
            "weight_decay_enc":2e-2, # fix
            "weight_decay_dec":5e-2, # fix
            "lr_enc":2e-3, # fix
            "lr":7e-4, # fix
            "num_epochs":300,
            "bs_enc":32, # fix
            "bs_dec":32 # fix
         },
     }

yelp={
         "model":{
            "enc_blocks":1, # fix
            "dec_blocks":3, 
            "n_experts":5,
            "topk":2,
            "num_heads":8,
            "d_model":64*8,
            "d_ffn":512,
            "dropout": 0.3,
            "moe":True
         },
         "training":{
            "weight_decay_enc":5e-2,
            "weight_decay_dec":9e-2,
            "lr_enc":5e-3,
            "lr":7e-4,
            "num_epochs":300,
            "bs_enc":32,
            "bs_dec":32
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