ciao_timestamp={  
         "model":{
            "enc_blocks":2,
            "dec_blocks":2,
            "num_heads":8,
            "n_experts":2,
            "topk":1,
            "d_model":128,
            "d_ffn":256,
            "dropout": 0.3,
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-3,
            "lr_enc":1e-2,
            "lr":5e-3,
            "num_epochs":300,
            "batch_size":64
         },
     }

epinions={
         "model":{
            "enc_blocks":2,
            "dec_blocks":2,
            "num_heads":4,
            "n_experts":4,
            "topk":2,
            "d_model":128,
            "d_ffn":256,
            "dropout": 0.3,
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-3,
            "lr_enc":1e-2,
            "lr":5e-3,
            "num_epochs":300,
            "batch_size":128
         },
     }

yelp={
         "model":{
            "enc_blocks":1,
            "dec_blocks":1,
            "num_heads":8,
            "n_experts":2,
            "topk":2,
            "user_seq_len":30,
            "item_seq_len":150,
            "d_model":128,
            "d_ffn":256,
            "dropout": 0.3,
         },
         "training":{
            "weight_decay_enc":5e-2,
            "weight_decay_dec":1e-2,
            "num_epochs":300,
            "lr":5e-3,
            "lr_enc":5e-3,
            "batch_size":128
         },
     }

Douban={
         "model":{
            "enc_blocks":1,
            "dec_blocks":1,
            "num_heads":8,
            "n_experts":2,
            "topk":2,
            "user_seq_len":30,
            "item_seq_len":150,
            "d_model":128,
            "d_ffn":256,
            "dropout": 0.3,
         },
         "training":{
            "weight_decay_enc":1e-1,
            "weight_decay_dec":1e-2,
            "num_epochs":300,
            "lr":5e-3,
            "lr_enc":1e-3,
            "batch_size":256
         },
     }



Config = {
    "ciao_timestamp":ciao_timestamp,
    "epinions":epinions,
    "yelp":yelp,
    "Douban":Douban
}