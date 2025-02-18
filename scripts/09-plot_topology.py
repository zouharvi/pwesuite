#!/usr/bin/env python3

import os
import argparse
import random
import panphon2
import numpy as np
from multiprocessing.pool import ThreadPool
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
# import main.fig_utils as fig_utils
import sys
sys.path.append(".")
from main.utils import load_multi_data, load_embd_data
import pickle

CLUSTER_SIZE = 4
CLUSTER_COUNT = 5

import matplotlib as mpl
from cycler import cycler
COLORS = [
    "cornflowerblue",
    "darkseagreen",
    "salmon",
    "orange",
    "purple",
    "dimgray",
    "seagreen",
]
mpl.rcParams['axes.prop_cycle'] = cycler(color=COLORS)

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
args.add_argument("--tsne-n", type=int, default=400)
args.add_argument("--plot-n", type=int, default=200)
args.add_argument("-s", "--seed", type=int, default=1)
args.add_argument("-ts", "--tsne-seed", type=int, default=1)
args = args.parse_args()

CACHE_PATH = "computed/cache/topology_data.pkl"
if os.path.exists(CACHE_PATH):
    data = pickle.load(open(CACHE_PATH, "rb"))
else:
    data_multi = load_multi_data(purpose_key="all")
    data_embd_1 = load_embd_data(args.embd_1)
    data_embd_2 = load_embd_data(args.embd_2)
    data_embd_3 = load_embd_data(args.embd_3)
    data = [
        (x, y)
        for x, y in zip(data_multi, zip(data_embd_1, data_embd_2, data_embd_3))
        if x["lang"] == "en"
    ]
    pickle.dump(data, open(CACHE_PATH, "wb"))

random.seed(args.seed)
data = random.sample(data, k=args.tsne_n)

def _compute_panphon_distance(y, data):
    # tok_features break pipe in multiprocess
    # TODO: multiprocessing?
    fed = panphon2.FeatureTable().feature_edit_distance
    return [fed(tok_ipa, y) for tok_ipa in data]

data_ipa = [x["token_ipa"] for x, y in data]
with ThreadPool() as pool:
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

plt.figure(figsize=(4, 1.8))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3)
# ax4 = plt.subplot(2, 2, 4)

def plot_scatter(ax, data_dists, title, flip_x=False, flip_y=False):
    print("Computing TSNE")
    model = TSNE(
        n_components=2, metric="precomputed",
        learning_rate="auto", init="random",
        random_state=args.tsne_seed,
        n_iter=2000,
    )
    data_2d = model.fit_transform(np.array(data_dists))
    data_2d -= np.mean(data_2d, axis=0)
    r = random.Random(args.seed)
    data_2d_plot_i = set(r.sample(range(len(data_2d)), k=args.plot_n))

    if flip_y:
        data_2d[:,1] = -data_2d[:,1]
    if flip_x:
        data_2d[:,0] = -data_2d[:,0]

    ax.scatter(
        [x[0] for i, x in enumerate(data_2d) if i in data_2d_plot_i],
        [x[1] for i, x in enumerate(data_2d) if i in data_2d_plot_i],
        color="tab:gray", s=12
    )
    for cluster in clusters:
        ax.scatter(
            [data_2d[i][0] for i in cluster],
            [data_2d[i][1] for i in cluster],
            s=18,
            linewidth=0,
        )

    ax.set_title(title)
    ax.axis('off')

    print(title)
    avg_dist_cluster_points = np.average([
        np.linalg.norm(data_2d[i_a]-data_2d[i_b])
        for cluster_a in clusters
        for cluster_b in clusters
        for i_a in cluster_a
        for i_b in cluster_b
    ])
    print("avg dist_cluster_points", avg_dist_cluster_points)

    avg_dist_within_cluster = np.average([
        np.average([
            np.linalg.norm(data_2d[i_a]-data_2d[i_b])
            for i_a in cluster_a
            for i_b in cluster_a
        ])
        for cluster_a in clusters
    ])
    print("avg dist within cluster", avg_dist_within_cluster)
    print("proportion * 100", avg_dist_within_cluster/avg_dist_cluster_points*100)

    ax.text(
        x=0.4*max(data_2d[:,0]),
        y=min(data_2d[:,1]),
        s=f"$d={avg_dist_within_cluster/avg_dist_cluster_points*100:.0f}$",
        ha="left",
    )


plot_scatter(ax1, data_dists_fed, "Art. Distance")
# t-SNE is flip-invariant, so we have to fix it manually
plot_scatter(ax2, data_dists_our1, "Art. Features", flip_x=True, flip_y=False)
# plot_scatter(ax3, data_dists_our2, "IPA Features")
plot_scatter(ax3, data_dists_our3, "Characters", flip_x=False, flip_y=True)

plt.tight_layout(pad=0.1)
plt.savefig("computed/figures/clusters.pdf")
plt.show()