#!/usr/bin/env python3

import math
import torch
import argparse
import tqdm
from model import RNNAutoencoder
import pickle
from models.metric_learning.preprocessor import preprocess_dataset

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-l", "--lang", default="en")
args.add_argument(
    "-mp", "--model-path",
    default="computed/models/rnn_autoencoder_en.pt"
)
args.add_argument("-o", "--output", default="computed/tmp/multi_en.pkl")
args.add_argument("--features", default="panphon")
args.add_argument("--dimension", type=int, default=300)
args = args.parse_args()

data = preprocess_dataset(args.data, args.features, args.lang, purpose_key="all")
BATCH_SIZE = 2000

print(f"Loaded {len(data)//1000}k words")

model = RNNAutoencoder(
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
