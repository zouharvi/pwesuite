#!/usr/bin/env python3

import panphon2
import numpy as np
import multiprocess as mp
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import argparse
from main.utils import load_embd_data, load_multi_data
import collections
import tqdm
import random

def evaluate_retrieval(data_multi_all, data_size=1000, jobs=20):
    data = []

    for (token_ort, token_ipa, lang, pronunciation, purpose, embd) in data_multi_all:
        if lang != "en":
            continue
        data.append((token_ipa, embd, token_ort))

    def compute_panphon_distance(y, data):
        fed = panphon2.FeatureTable().feature_edit_distance
        return [fed(w, y) for w, _, _ in data]

    # Take only dev data
    r = random.Random(0)
    data = r.sample(data, k=data_size)

    with mp.Pool(20) as pool:
        data_dists_fed = np.array(pool.map(
            lambda y: compute_panphon_distance(y[0], data),
            data
        ))

    data_dists_l2 = euclidean_distances(np.array([x[1] for x in data]))
    data_dists_cos = cosine_distances(np.array([x[1] for x in data]))

    for word_i, (dist_fed, dist_l2, dist_cos) in enumerate(zip(data_dists_fed, data_dists_l2, data_dists_cos)):
        # get neighbour indicies
        order_fed = [
            x[0] for x
            in sorted(enumerate(dist_fed), key=lambda x: x[1])
        ]
        order_l2 = [
            x[0] for x
            in sorted(enumerate(dist_l2), key=lambda x: x[1])
        ]
        order_cos = [
            x[0] for x
            in sorted(enumerate(dist_cos), key=lambda x: x[1])
        ]

        # skip self
        nearest_fed = order_fed[1]
        nearest_in_l2 = order_l2.index(nearest_fed)
        nearest_in_cos = order_cos.index(nearest_fed)

        word_orig = data[word_i][2]
        word_neighbour = data[nearest_fed][2]
        word_retrieved = data[order_l2[1]][2]
        if word_orig[0] == word_neighbour[0] and word_orig[0] != word_retrieved[0] and word_neighbour != word_retrieved:
            print(word_orig, word_neighbour, " ||| ", word_retrieved)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-e", "--embd", default="computed/embd_bpemb.pkl")
    args = args.parse_args()

    data_embd = load_embd_data(args.embd)
    data_multi_all = load_multi_data(purpose_key="all")

    data_multi = [
        (*x, np.array(y)) for x, y in zip(data_multi_all, data_embd)
        if x[3] == "main"
    ]

    output = evaluate_retrieval(data_multi)