#!/usr/bin/env python3

import math
import panphon2
import torch
import argparse
import tqdm
from models.metric_learning.model import RNNMetricLearner
import pickle
from main.utils import load_multi_data

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-l", "--lang", default="en")
args.add_argument(
    "-mp", "--model-path",
    default="computed/models/rnn_metric_learning_en.pt"
)
args.add_argument("-o", "--output", default="computed/tmp/multi_en.pkl")
args = args.parse_args()

data = [
    x[1] for x in load_multi_data(args.data)
    if x[2] == args.lang
]

BATCH_SIZE = 1000

print(f"Loaded {len(data)//1000}k words")

f = panphon2.FeatureTable()
data = [(w, f.word_to_binary_vectors(w)) for w in tqdm.tqdm(data)]

# target metric here doesn't play a role
model = RNNMetricLearner(target_metric="l2")
model.load_state_dict(torch.load(args.model_path))
model.eval()

# some cheap paralelization
data_out = []
for i in tqdm.tqdm(range(math.ceil(len(data)/BATCH_SIZE))):
    batch = [b for _, b in data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
    data_out += list(
        model.forward(batch).detach().cpu().numpy()
    )

print("Embedding dimension:", data_out[-1].shape)
assert len(data) == len(data_out)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)