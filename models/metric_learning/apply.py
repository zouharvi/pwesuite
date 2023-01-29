#!/usr/bin/env python3

import math
import torch
import argparse
import tqdm
from models.metric_learning.model import RNNMetricLearner
import pickle
from preprocessor import preprocess_dataset
from main.utils import load_multi_data

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-l", "--lang", default="en")
args.add_argument(
    "-mp", "--model-path",
    default="computed/models/rnn_metric_learning_en.pt"
)
args.add_argument("-o", "--output", default="computed/tmp/multi_en.pkl")
args.add_argument("--features", default="panphon")
args.add_argument("--dimension", type=int, default=300)

args = args.parse_args()

# token_ort, token_ipa, lang, pronunc
data = [
    x[:2] for x in load_multi_data(args.data)
    if x[2] == args.lang
][:1000 + args.number_thousands * 1000]

data = preprocess_dataset(data, args.features)
BATCH_SIZE = 1000

print(f"Loaded {len(data)//1000}k words")

# target metric here doesn't play a role
model = RNNMetricLearner(
    target_metric="l2",
    dimension=args.dimension,
    feature_size=data[0][0].shape[1],
)
model.load_state_dict(torch.load(args.model_path))
model.eval()

# some cheap paralelization
data_out = []
for i in tqdm.tqdm(range(math.ceil(len(data) / BATCH_SIZE))):
    batch = [f for f, _ in data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
    data_out += list(
        model.forward(batch).detach().cpu().numpy()
    )

assert len(data) == len(data_out)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)
