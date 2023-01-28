#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import torch
import panphon2
from models.metric_learning.rnn_metric_learning_model import RNNMetricLearner
from sklearn.metrics.pairwise import euclidean_distances

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/embds_pl.pkl")
args.add_argument("-m", "--model", default="computed/models/model_pl.pt")
args.add_argument("-n", "--n", type=int, default=1000)
args = args.parse_args()
ft = panphon2.FeatureTable()

with open(args.input, "rb") as f:
    data = pickle.load(f)

# use train data
data = data[1000:1000 + args.n]

model = RNNMetricLearner(target_metric="l2")
model.load_state_dict(torch.load(args.model))

ANALOG_WORDS = [
    "zadba", "ʐadba", # A, B
    "sadba", "ʂadba", # C, D
]
# A-B = C-D
# D = C-A+B
# find similarities to D

def set_similarity(a, b):
    return 2 * len(a.intersection(b)) / (len(a) + len(b))

set_D = set(ANALOG_WORDS[3])
# take top 20 most similar ones
data_hay = sorted(data, key=lambda x: set_similarity(set(x[0]), set_D), reverse=True)[:20]

# process analog words
data_analog = [(w, ft.word_to_binary_vectors(w)) for w in ANALOG_WORDS]
data_analog = [
    (w, b, model.forward([b])[0].detach().cpu().numpy())
    for w,b in data_analog
]
needle = data_analog[2][2]-data_analog[0][2]+data_analog[1][2]

# add D to search data
data_hay += [data_analog[3]]

data_hay = [
    (x[0], x[1], x[2], euclidean_distances([needle], [x[2]])[0,0])
    for x in data_hay
]

print(f"Searching for v({ANALOG_WORDS[2]})-v({ANALOG_WORDS[0]})+v({ANALOG_WORDS[1]})")
print(f"The correct word should be {ANALOG_WORDS[3]}")
data_hay.sort(key=lambda x: x[3])
for word, _bin_word, _vec, sim in data_hay:
    extra = "*" if word == ANALOG_WORDS[3] else " "
    print(f"{extra}{word:>15}: {sim:.5f}")