#!/usr/bin/env python3

import numpy as np
import argparse
import collections
import tqdm
import re
from main.utils import load_multi_data, load_embd_data
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import random

# TODO: cache this?
def evaluate_rhyme(data_multi_all):
    # token_ort, token_ipa, lang, purpose, token_pron, embd
    data_multi = [
        # embd, token_ort, token_ipa, token_pron
        (embd, x["token_ort"], x["token_ipa"], x["token_arp"]) for x, embd in data_multi_all
        # we have pronunciation information only for English
        if x["lang"] == "en"
    ]

    RE_LAST_STRESSED = re.compile(r".* ([\w]+1.*)")
    rhyme_clusters = collections.defaultdict(list)

    # rules for determining what's a ryme and not
    for embd, token_ort, token_ipa, token_arp in tqdm.tqdm(data_multi):
        if "1" not in token_arp:
            continue
        rhyme_part = RE_LAST_STRESSED.match(" " + token_arp).group(1)

        # all the embeddings within one cluster should rhyme
        # TODO: change this to different rhyme patterns
        rhyme_clusters[rhyme_part].append(embd)

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

    data_dev, data_train = train_test_split(data_task, test_size=1000, random_state=0)

    model = MLPClassifier(
        hidden_layer_sizes=(50, 20, 10),
        random_state=0,
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

    return {"train": max(acc_train, 1-acc_train), "dev": max(acc_dev, 1-acc_dev)}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-e", "--embd", default="computed/embd_bert.pkl")
    args = args.parse_args()

    data_embd = load_embd_data(args.embd)
    data_multi_all = load_multi_data(purpose_key="all")

    data_multi = [
        (x, np.array(y)) for x, y in zip(data_multi_all, data_embd)
        if x["purpose"] == "main"
    ]

    output = evaluate_rhyme(data_multi)

    print("Overall:")
    for key in output:
        print(f"{key}: {output[key]:.1%}")