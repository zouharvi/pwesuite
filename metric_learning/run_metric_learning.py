#!/usr/bin/env python3

import panphon2
import argparse
import tqdm
from metric_learning.rnn_metric_learning_model import RNNMetricLearner

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/ipa_tokens_en.txt")
args.add_argument(
    "-nk", "--number-thousands", default=99, type=int,
    help="Number of training data to use (in thousands) for training",
)
args.add_argument(
    "--eval-train-full", action="store_true",
    help="Compute correlations also for full the training data instead of just 1k sample. This will significantly slower your code."
)
args.add_argument("-tm", "--target-metric", default="l2")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [x.rstrip("\n") for x in f.readlines()][:1000+args.number_thousands*1000]

print(f"Loaded {len(data)//1000}k words")

f = panphon2.FeatureTable()
data = [(w, f.word_to_binary_vectors(w)) for w in tqdm.tqdm(data)]

data_dev = data[:1000]
data_train = data[1000:]

# target_metric="ip" is not good for some reason
model = RNNMetricLearner(target_metric=args.target_metric)
model.train_epochs(data_train, data_dev, eval_train_full=args.eval_train_full)
