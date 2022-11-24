#!/usr/bin/env python3

import panphon2
import argparse
import tqdm
import pickle
import torch
from rnn_metric_learner import RNNMetricLearner

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/ipa_tokens_en.txt")
args.add_argument("-v", "--vocab", default="data/vocab_en.txt")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [x.rstrip("\n") for x in f.readlines()][:5000]

print(f"Loaded {len(data)//1000}k words")

f = panphon2.FeatureTable()
data = [(w, f.word_to_binary_vectors(w)) for w in tqdm.tqdm(data)]

data_dev = data[:1000]
data_train = data[1000:]

# fed = 

model = RNNMetricLearner(target_metric="l2")
model.train_epochs(data_train, data_dev)