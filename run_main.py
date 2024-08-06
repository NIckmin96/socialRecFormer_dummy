import subprocess
import requests
import json
from datetime import datetime
import os
import torch

seed = ["42"]
data_seed = ["42"]
datasets = ['ciao']
#lrs = ['5e-5']
lrs = ['1e-4']
enc_layers = ['2']
dec_layers = ['2']
#python3 data_making.py --dataset "$DATASET" --first True --seed 42 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"
user_item = [(30,100)]

rps = ['1']
for s in range(50):
    for ds in data_seed:
        s = str(s)
        for lr in lrs:
            for el in enc_layers:
                #for dl in dec_layers:
                    for d in datasets:
                        for (u,i) in user_item:
                                for rp in rps:
                                    name = f'{u}-{i}-{lr}-enc{el}-dec{el}_rp{rp}_5X_b128'
                                    path = f"./logs/log_seed_{s}/{d}/train.{name}.log"
                                    print(name)
                                    if not os.path.isfile(path):
                                        torch.cuda.empty_cache()
                                        subprocess.run(['python', 'main.py', '--seed', s, '--data_seed', ds, '--dataset', d, '--user_seq_len', str(u), '--item_seq_len', str(i), '--name', name, '--lr', lr, '--num_layers_enc', el, '--num_layers_dec', el, '--return_params', rp])
                                        torch.cuda.empty_cache()
                                    else:
                                        print(f"Exists!")
