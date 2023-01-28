#!/usr/bin/env python3

import panphon2
import torch
import argparse
import tqdm
from models.metric_learning.model import RNNMetricLearner
from main.utils import load_multi_data

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-l", "--lang", default="en")
args.add_argument(
    "-smp", "--save-model-path",
    default="computed/models/rnn_metric_learning_en.pt"
)
args.add_argument("-e", "--epochs", type=int, default=20)
args.add_argument(
    "-nk", "--number-thousands", type=int, default=200,
    help="Number of training data to use (in thousands) for training",
)
args.add_argument(
    "--eval-train-full", action="store_true",
    help="Compute correlations also for full the training data instead of just 1k sample. This will be significantly slower."
)
args.add_argument("-tm", "--target-metric", default="l2")
args.add_argument("--dimension", type=int, default=300)
args = args.parse_args()

data = [
    x[1] for x in load_multi_data(args.data)
    if x[2] == args.lang
][:1000 + args.number_thousands * 1000]

print(f"Loaded {len(data)//1000}k words")

f = panphon2.FeatureTable()
data = [(w, f.word_to_binary_vectors(w)) for w in tqdm.tqdm(data)]

data_dev = data[:1000]
data_train = data[1000:]

# target_metric="ip" is not good for some reason
model = RNNMetricLearner(
    target_metric=args.target_metric,
    dimension=args.dimension,
)
model.train_epochs(
    data_train=data_train, data_dev=data_dev,
    eval_train_full=args.eval_train_full,
    epochs=args.epochs
)

if args.save_model_path is not None:
    torch.save(model.state_dict(), args.save_model_path)