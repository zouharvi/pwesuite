#!/usr/bin/env python3

import panphon2
import numpy as np
import multiprocess as mp
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import argparse
from main.utils import load_embd_data, load_multi_data
import collections
import tqdm

def evaluate_retrieval(data_multi, data_embd, data_size=1000, jobs=20):
    data_langs = collections.defaultdict(list)

    for (token_ort, token_ipa, lang, pronunciation), emdb in zip(data_multi, data_embd):
        data_langs[lang].append((token_ipa, emdb))

    def compute_panphon_distance(y, data):
        fed = panphon2.FeatureTable().feature_edit_distance
        return [fed(w, y) for w, _ in data]

    rank_l2_all = {}
    rank_cos_all = {}

    for lang, data in tqdm.tqdm(data_langs.items()):
        # Take only dev data
        data = data[:data_size]

        with mp.Pool(20) as pool:
            data_dists_fed = np.array(pool.map(
                lambda y: compute_panphon_distance(y[0], data),
                data
            ))

        data_dists_l2 = euclidean_distances(np.array([x[1] for x in data]))
        data_dists_cos = cosine_distances(np.array([x[1] for x in data]))

        rank_l2 = []
        rank_cos = []

        for dist_fed, dist_l2, dist_cos in zip(data_dists_fed, data_dists_l2, data_dists_cos):
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

            rank_l2.append((data_size-nearest_in_l2)/data_size)
            rank_cos.append((data_size-nearest_in_cos)/data_size)

        rank_l2_all[lang] = np.average(rank_l2)
        rank_cos_all[lang] = np.average(rank_cos)

    rank_l2_all["all"] = np.average(list(rank_l2_all.values()))
    rank_cos_all["all"] = np.average(list(rank_cos_all.values()))


    return {
        "rank L2": rank_l2_all, "rank cos": rank_cos_all,
    }

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data-multi", default="data/multi.tsv")
    args.add_argument("-e", "--embd", default="computed/embd_bpemb.pkl")
    args = args.parse_args()

    data_multi = load_multi_data(args.data_multi)
    data_embd = load_embd_data(args.embd)

    assert len(data_multi) == len(data_embd)

    output = evaluate_retrieval(data_multi, data_embd)

    print("Overall:")
    for key in output:
        print(f"{key}: {output[key]['all']:.3f}")