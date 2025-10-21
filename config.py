ciao_timestamp={  
         "model":{
            "enc_blocks":2,
            "dec_blocks":2,
            "num_heads":6,
            "n_experts":6,
            "topk":3,
            "d_model":32*6,
            "d_ffn":256,
            "dropout": 0.3,
            "moe":True
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-2,
            "lr_enc":1e-2,
            "lr":1e-3,
            "num_epochs":300,
            "bs_enc":32,
            "bs_dec":64
         },
     }

# epinions={
#          "model":{
#             "enc_blocks":2,
#             "dec_blocks":2,
#             "num_heads":6,
#             "n_experts":6,
#             "topk":2,
#             "d_model":32*6,
#             "d_ffn":256,
#             "dropout": 0.3,
#             "moe":True
#          },
#          "training":{
#             "weight_decay_enc":1e-2,
#             "weight_decay_dec":1e-3,
#             "lr_enc":1e-2,
#             "lr":1e-3,
#             "num_epochs":300,
#             "bs_enc":128,
#             "bs_dec":128
#          },
#      }

epinions={
         "model":{
            "enc_blocks":1,
            "dec_blocks":2,
            "num_heads":8,
            "n_experts":2,
            "topk":2,
            "d_model":128,
            "d_ffn":256,
            "dropout": 0.3,
            "moe":True
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-3,
            "lr_enc":7e-3,
            "lr":4e-3,
            "num_epochs":300,
            "bs_enc":32,
            "bs_dec":64
         },
     }

yelp={
         "model":{
            "enc_blocks":1,
            "dec_blocks":2,
            "num_heads":8,
            "n_experts":3,
            "topk":1,
            "d_model":128,
            "d_ffn":256,
            "dropout": 0.3,
            "moe":True
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-2,
            "lr_enc":7e-3,
            "lr":4e-3,
            "num_epochs":300,
            "bs_enc":128,
            "bs_dec":128,
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