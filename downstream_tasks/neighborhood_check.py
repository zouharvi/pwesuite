#!/usr/bin/env python3

import argparse
import panphon2
import pickle
import numpy as np
import multiprocess as mp
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import fig_utils

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/embds_en.pkl")
args.add_argument("-n", "--n", type=int, default=1000)
args = args.parse_args()

with open(args.input, "rb") as f:
    data = pickle.load(f)

# use train data
data = data[1000:1000+args.n]

def compute_panphon_distance(y, data):
    fed = panphon2.FeatureTable().feature_edit_distance
    return [fed(w, y) for w, _, _ in data]

print("Computing Panphon FED")
with mp.Pool() as pool:
    data_dists_fed = np.array(pool.map(
        lambda y: compute_panphon_distance(y[0], data),
        data
    ))

print("Computing cosine similarities")
data_dists_cos = euclidean_distances(np.array([x[2] for x in data]))

data_dists_cos_flat = [x for y in data_dists_cos for x in y]
data_dists_fed_flat = [x for y in data_dists_fed for x in y]
corr_pearson = pearsonr(data_dists_cos_flat, data_dists_fed_flat)
corr_spearman = spearmanr(data_dists_cos_flat, data_dists_fed_flat)
print(f"Pearson: {corr_pearson[0]:.2%}")
print(f"Spearman: {corr_spearman[0]:.2%}")

# get nearest neighbour indicies
data_dists_fed = [
    [x[0] for x in sorted(list(enumerate(row)), key=lambda x: x[1])]
    for row in data_dists_fed
]
data_dists_cos = [
    [x[0] for x in sorted(list(enumerate(row)), key=lambda x: x[1])]
    for row in data_dists_cos
]

Ks = list(range(1, 50))
data_overlap = [
    np.average([
        x[1] in y[1:k+1]
        for x, y in zip(data_dists_fed, data_dists_cos)
    ])
    for k in Ks
]

plt.plot(
    Ks, data_overlap,
    marker=".",
)
plt.plot(
    [20], [data_overlap[19]],
    color=fig_utils.COLORS_EXTRA[0],
    marker=".",
    markersize=20,
)
plt.plot(
    [1], [data_overlap[0]],
    color=fig_utils.COLORS_EXTRA[1],
    marker=".",
    markersize=20,
)
plt.ylim(0, 1)
plt.text(
    x=20, y=data_overlap[19]+0.1,
    s=f"On average, the closest neighbour to a point in FED is in the\n20 neighbours of COS space {data_overlap[19]:.0%} of times",
    ha="center",
    color=fig_utils.COLORS_EXTRA[0],
)
plt.text(
    x=1+1, y=data_overlap[0],
    s=f"Only in {data_overlap[0]:.0%} cases is the nearest neighbour the same.",
    ha="left",
    color=fig_utils.COLORS_EXTRA[1],
)

plt.xlabel("Neighbourhood size")
plt.ylabel("Is closets neighbour of point in FED in top-K\nneighbours of COS? (Retrieval over 1k examples)")
plt.tight_layout()
plt.show()
