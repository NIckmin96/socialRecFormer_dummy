ciao_timestamp={  
         # dataset의 전체 길이 (기록 및 확인용) (train:val:test = 8:1:1)
         "model":{
            "num_heads":2,
            "enc_blocks":1,
            "dec_blocks":1,
            "dropout": 0.3,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
         },
         "training":{
            "weight_decay":1e-2,
            "num_epochs":300,
            "lr":5e-3,
            "lr_enc":8e-3
         },
     }

epinions={
         "model":{
            "num_heads":2,
            "enc_blocks":1,
            "dec_blocks":1,
            "dropout": 0.3,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
         },
         "training":{
            "weight_decay":1e-2,
            "num_epochs":300,
            "lr":5e-3,
            "lr_enc":5e-3
         },
     }

yelp={
         "model":{
            "num_heads": 2,
            "dropout": 0.3,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
         },
         "training":{
            "weight_decay":1e-2,
            "num_epochs":300,
         },
     }

Douban={
         "model":{
            "num_heads": 2,
            "dropout": 0.3,         # Inside FFN, decoder_layer & encoder_layer (applied after linear & attention)
         },

         "training":{
            "weight_decay":1e-2,
            "num_epochs":300
         },
     }



Config = {
    "ciao_timestamp":ciao_timestamp,
    "epinions":epinions,
    "yelp":yelp,
    "Douban":Douban
}