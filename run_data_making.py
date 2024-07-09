import subprocess
import requests
import json
from datetime import datetime
import os
import torch

seed = ["42"]
datasets = ['ciao']
user_item = [(30,100)]

#python3 data_making.py --dataset "$DATASET" --first True --seed 42 --test_ratio 0.1 --random_walk_len "$RANDOM_WALK_LEN" --item_seq_len "$ITEM_SEQ_LEN"
rps = ['1']
for s in seed:
    for d in datasets:
        for (u, i) in user_item:
            for rp in rps:
                path = f"./dataset/{d}/sequence_data_seed_{s}_walk_{u}_itemlen_{i}_rp_{rp}_.pkl"
                print(path)
                if not os.path.isfile(path):
                    torch.cuda.empty_cache()
                    subprocess.run(['python', 'data_making.py', '--seed', s, '--dataset', d, '--random_walk_len', str(u), '--item_seq_len', str(i), '--return_params', rp])
                    torch.cuda.empty_cache()
                else:
                    print(f"Exists!")
