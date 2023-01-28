#!/usr/bin/env python3

import random
import torch
import numpy as np
import pickle
import argparse
import tqdm
import spacy
from models.metric_learning.rnn_metric_learning_model import RNNMetricLearner
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

args = argparse.ArgumentParser()
# make sure it's re-saved with UTF-8, otherwise it will cause issues
args.add_argument("-mi", "--model-input", default="computed/model.pt")
args.add_argument("-i", "--input", default="data/rhymes.pkl")
args = args.parse_args()

# load dataset
with open(args.input, "rb") as f:
    rhyme_clusters = list(pickle.load(f).values())

# load model
model = RNNMetricLearner(target_metric="l2")
model.load_state_dict(torch.load(args.model_input))

# create task
random.seed(0)
data_task = []
for i in range(20000 // 2):
    # select two clusters
    cluster_a, cluster_b = random.sample(rhyme_clusters, k=2)
    # select random element from those clusters
    el_orig = random.choice(cluster_a)
    el_pos = random.choice(cluster_a)
    el_neg = random.choice(cluster_b)
    data_task.append((True, (el_orig, el_pos)))
    data_task.append((False, (el_orig, el_neg)))

print(len(data_task))

# process data
nlp = spacy.load("en_core_web_lg")

data_task = [
    (y, np.concatenate((
        nlp(x1).vector,
        nlp(x2).vector
    ), axis=0)
    ) for y, (x1, x2) in tqdm.tqdm(data_task)
]

# make split
data_dev = data_task[:1000]
data_train = data_task[1000:]
print(len(data_dev), len(data_train))
print(data_dev[0][1].shape, "vector shape")

model = MLPClassifier(
    hidden_layer_sizes=(50, 20, 10),
)
model.fit(
    [x[1] for x in data_train],
    [x[0] for x in data_train],
)
acc_train = model.score(
    [x[1] for x in data_train],
    [x[0] for x in data_train],
)
acc_dev = model.score(
    [x[1] for x in data_dev],
    [x[0] for x in data_dev],
)
print(f"Train: {acc_train:.2%}")
print(f"Dev: {acc_dev:.2%}")
