#!/usr/bin/env python3

import numpy as np
import argparse
import collections
import tqdm
import re
from main.utils import load_multi_data, load_embd_data
from sklearn.neural_network import MLPClassifier
import random

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-e", "--embd", default="computed/embd_bert.pkl")
args = args.parse_args()


# This can't be unfortunately cached because of the zip here
data = load_multi_data(args.data)
data_embd = load_embd_data(args.embd)

data = [
    (embd, x[0], x[1], x[3]) for embd, x in zip(data_embd, data)
    # we have pronunciation information only for English
    if x[2] == "en"
]
print(f"Loaded {len(data)//1000}k words")

RE_LAST_STRESSED = re.compile(r".* ([\w]+1.*)")
rhyme_clusters = collections.defaultdict(list)

# rules for determining what's a ryme and not
for embd, token_ort, token_ipa, pronunc in tqdm.tqdm(data):
    if "1" not in pronunc:
        continue
    rhyme_part = RE_LAST_STRESSED.match(" " + pronunc).group(1)

    # all the embeddings within one cluster should rhyme
    # TODO: change this to different rhyme patterns
    rhyme_clusters[rhyme_part].append(embd)

print(len(rhyme_clusters), "rhyme clusters")
print(f"average cluster size: {np.average([len(x) for x in rhyme_clusters.values()]):.1f}")

random.seed(0)
rhyme_part_keys = list(rhyme_clusters.keys())
data_task = []
for rhyme_part, cluster in rhyme_clusters.items():
    if len(cluster) < 2:
        continue
    embd1, embd2 = random.sample(cluster, k=2)
    while True:
        key = random.choice(rhyme_part_keys)
        if key == rhyme_part:
            continue
        embd3 = random.choice(rhyme_clusters[key])
        break

    # make sure everything is numpy
    embd1 = np.array(embd1)
    embd2 = np.array(embd2)
    embd3 = np.array(embd3)

    data_task.append((np.concatenate((embd1, embd2)), True))
    data_task.append((np.concatenate((embd1, embd3)), False))

data_dev = data_task[:1000]
data_train = data_task[1000:]
print("Dev size:", len(data_dev))
print("Train size:", len(data_train))

model = MLPClassifier(
    hidden_layer_sizes=(50, 20, 10),
)
model.fit(
    [x[0] for x in data_train],
    [x[1] for x in data_train],
)
acc_train = model.score(
    [x[0] for x in data_train],
    [x[1] for x in data_train],
)
acc_dev = model.score(
    [x[0] for x in data_dev],
    [x[1] for x in data_dev],
)
print(f"Train: {acc_train:.1%}")
print(f"Dev: {acc_dev:.1%}")
