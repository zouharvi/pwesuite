#!/usr/bin/env python3

import panphon2
import torch
import argparse
import tqdm
from rnn_metric_learning_model import RNNMetricLearner
import pickle

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/ipa_tokens_pl.txt")
args.add_argument("-o", "--output", default="computed/embds_pl.pkl")
args.add_argument("-smp", "--save-model-path", default="models/model_pl.pt")
args.add_argument("-e", "--epochs", type=int, default=20)
args.add_argument(
    "-nk", "--number-thousands", type=int, default=99,
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
model.train_epochs(data_train, data_dev, eval_train_full=args.eval_train_full, epochs=args.epochs)

if args.output is not None:
    # TODO: paralelize
    model.eval()
    data = [(w, b, model.forward([b])[0].detach().cpu().tolist()) for w, b in tqdm.tqdm(data)]
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

if args.save_model_path is not None:
    torch.save(model.state_dict(), args.save_model_path)