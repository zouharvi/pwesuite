#!/usr/bin/env python3

import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import argparse
from scipy.stats import pearsonr, spearmanr
from main.utils import load_embd_data, load_multi_data
import collections


def evaluate_human_similarity(data_multi_hs):
    tok_to_embd = {}
    for (token_ort, token_ipa, lang, pronunciation, purpose, embd) in data_multi_hs:
        tok_to_embd[token_ort] = embd

    with open("data/human_similarity.csv", "r") as f:
        data_hs = list(csv.DictReader(f))

    batches = collections.defaultdict(list)
    for w_hs in data_hs:
        batches[w_hs["word2"]].append((
            tok_to_embd[w_hs["word1"]],
            tok_to_embd[w_hs["word2"]],
            w_hs["obtained"],
        ))
    corr_pearson_l2_all = []
    corr_spearman_l2_all = []
    corr_pearson_cos_all = []
    corr_spearman_cos_all = []
    for batch in batches.values():
        predicted_cos = [
            cosine_distances([e1], [e2])[0,0]
            for e1, e2, _ in batch
        ]
        predicted_l2 = [
            -euclidean_distances([e1], [e2])[0,0]
            for e1, e2, _ in batch
        ]
        obtained = [float(o) for _, _, o in batch]

        corr_pearson_l2_all.append(pearsonr(predicted_l2, obtained)[0])
        corr_pearson_cos_all.append(pearsonr(predicted_cos, obtained)[0])
        corr_spearman_l2_all.append(spearmanr(predicted_l2, obtained)[0])
        corr_spearman_cos_all.append(spearmanr(predicted_cos, obtained)[0])

    return {
        "pearson L2": abs(np.average(corr_pearson_l2_all)),
        "pearson cos": abs(np.average(corr_pearson_cos_all)),
        "spearman L2": abs(np.average(corr_spearman_l2_all)),
        "spearman cos": abs(np.average(corr_spearman_cos_all)),
    }

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data-multi", default="data/multi.tsv")
    args.add_argument("-e", "--embd", default="computed/embd_bpemb.pkl")
    args = args.parse_args()

    data_embd = load_embd_data(args.embd)
    data_multi_all = load_multi_data(args.data_multi, purpose_key="all")

    data_multi = [
        (*x, y) for x, y in zip(data_multi_all, data_embd)
        if x[3] == "human_similarity"
    ]

    output = evaluate_human_similarity(data_multi)
    print("Overall:")
    for key in output:
        print(f"{key}: {output[key]:.2f}")
