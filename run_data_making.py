import subprocess

# Standalone data-prep runner. Matches data_making.py::_get_args (the unified pipeline
# that main.py also drives via dm.DatasetMaking).
seeds = ["42"]
datasets = ["ciao"]
# (user_seq_len = random walk length, item_per_user)
user_item = [(30, 5)]
augs = "1"
regen = "all"          # no | all | rw | total | train

for s in seeds:
    for d in datasets:
        for (usl, ipu) in user_item:
            cmd = [
                "python", "data_making.py",
                "--dataset", d,
                "--seed", s,
                "--regen", regen,
                "--user_seq_len", str(usl),
                "--item_per_user", str(ipu),
                "--augs", augs,
                # "--neg",            # uncomment to add negatives to the train target set
                # "--drop_cold_eval", # uncomment to exclude cold (0 train interaction) anchors from valid/test
            ]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
