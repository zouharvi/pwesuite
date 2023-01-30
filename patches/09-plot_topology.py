#!/usr/bin/env python3

import argparse
import random
import panphon2
import numpy as np
import multiprocess as mp
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import main.fig_utils as fig_utils
from main.utils import load_multi_data, load_embd_data

CLUSTER_SIZE = 4
CLUSTER_COUNT = 5

args = argparse.ArgumentParser()
args.add_argument(
    "-e1", "--embd-1",
    default="computed/embd_rnn_metric_learning/panphon.pkl"
)
args.add_argument(
    "-e2", "--embd-2",
    default="computed/embd_rnn_metric_learning/tokenipa.pkl"
)
args.add_argument(
    "-e3", "--embd-3",
    default="computed/embd_rnn_metric_learning/tokenort.pkl"
)
args.add_argument("--tsne-n", type=int, default=2000)
args.add_argument("--plot-n", type=int, default=10)
args.add_argument("-s", "--seed", type=int, default=0)
args = args.parse_args()

data_multi = load_multi_data()
data_embd_1 = load_embd_data(args.embd_1)
data_embd_2 = load_embd_data(args.embd_2)
data_embd_3 = load_embd_data(args.embd_3)
data = [
    (x, y)
    for x, y in zip(data_multi, zip(data_embd_1, data_embd_2, data_embd_3))
    if x[2] == "en"
]

random.seed(1)
data = random.sample(data, k=300)

def _compute_panphon_distance(y, data):
    # tok_features break pipe in multiprocess
    fed = panphon2.FeatureTable().feature_edit_distance
    return [fed(tok_ipa, y) for tok_ipa in data]

data_ipa = [x[1] for x, y in data]
with mp.Pool() as pool:
    data_dists_fed = list(pool.map(
        lambda y: (
            _compute_panphon_distance(y, data_ipa)
        ),
        data_ipa
    ))

closest_points = sorted(
    list(range(len(data_dists_fed))),
    key=lambda i: sum(sorted(data_dists_fed[i])[:CLUSTER_SIZE])
)

banlist = set()
clusters = []
for point_i in closest_points:
    closest_i = sorted(
        range(len(data_dists_fed[0])),
        key=lambda i: data_dists_fed[point_i][i]
    )[:CLUSTER_SIZE*2]
    # block even a bit further points
    
    if len(banlist & set(closest_i)) != 0:
        continue

    banlist |= set(closest_i)
    clusters.append(set(closest_i[:CLUSTER_SIZE]))
    print("ADDING", point_i, closest_i, sum(data_dists_fed[point_i][i] for i in closest_i))
    if len(clusters) == CLUSTER_COUNT:
        break


data_dists_our1 = euclidean_distances(np.array([y[0] for x, y in data]))
data_dists_our2 = euclidean_distances(np.array([y[1] for x, y in data]))
data_dists_our3 = euclidean_distances(np.array([y[2] for x, y in data]))

plt.figure(figsize=(4, 4))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

def plot_scatter(ax, data_dists, title):
    print("Computing TSNE")
    model = TSNE(
        n_components=2, metric="precomputed",
        learning_rate="auto", init="random",
        random_state=1,
    )
    data_2d = model.fit_transform(np.array(data_dists))
    data_2d -= np.mean(data_2d, axis=0)

    ax.scatter(
        [x[0] for x in data_2d],
        [x[1] for x in data_2d],
        color="tab:gray", s=12
    )
    for cluster in clusters:
        ax.scatter(
            [data_2d[i][0] for i in cluster],
            [data_2d[i][1] for i in cluster],
            s=18,
        )

    ax.set_title(title)
    ax.axis('off')

plot_scatter(ax1, data_dists_fed, "Feature Edit Distance")
plot_scatter(ax2, data_dists_our1, "Panphon Features")
plot_scatter(ax3, data_dists_our2, "IPA Features")
plot_scatter(ax4, data_dists_our3, "Character Features")

plt.tight_layout(pad=0.1)
plt.savefig("computed/figures/clusters.pdf")
plt.show()