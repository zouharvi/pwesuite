#!/usr/bin/env python3

import argparse
import random
import panphon2
import pickle
from multiprocessing.pool import ThreadPool
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import main.fig_utils as fig_utils

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="computed/embds_en.pkl")
args.add_argument("--tsne-n", type=int, default=2000)
args.add_argument("--plot-n", type=int, default=10)
args.add_argument("-s", "--seed", type=int, default=0)
args = args.parse_args()

with open(args.input, "rb") as f:
    data = pickle.load(f)

print("Loaded", len(data), "but using", args.tsne_n)
data = data[:args.tsne_n]

def plot_data(ax, i, data_2d, label):
    ax.scatter(
        [x[0][0] for x in data_2d],
        [x[0][1] for x in data_2d],
        color=fig_utils.COLORS_EXTRA[i],
    )
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])

    data_2d_points = np.array([x[0] for x in data_2d])

    for x_i, (x_2d, x) in enumerate(data_2d):
        ax.text(
            x=x_2d[0], y=x_2d[1],
            s=x[0],
            va="bottom", ha="center",
        )

        # draw line to nearest neighbour
        dists = np.linalg.norm(np.array(x_2d)-data_2d_points, axis=1)
        # set infinity from self
        dists[x_i] = np.inf

        x_2d_closest = data_2d_points[np.argmin(dists)]

        ax.plot(
            [x_2d[0], x_2d_closest[0]],
            [x_2d[1], x_2d_closest[1]],
            color="black",
        )


def compute_panphon_distance(y, data):
    fed = panphon2.FeatureTable().feature_edit_distance
    return [fed(w, y) for w, _, _ in data]


print("Computing Panphon FED")
with ThreadPool() as pool:
    data_dists_fed = np.array(pool.map(
        lambda y: compute_panphon_distance(y[0], data),
        data
    ))

print("Computing TSNE")
model_fed = TSNE(
    n_components=2, metric="precomputed",
    learning_rate="auto", init="random"
)
data_2d_fed = model_fed.fit_transform(data_dists_fed)
data_2d_fed -= np.mean(data_2d_fed, axis=0)

print("Computing cosine similarities")
data_dists_cos = cosine_distances(np.array([x[2] for x in data]))

print("Computing TSNE")
model_cos = TSNE(
    n_components=2, metric="precomputed",
    learning_rate="auto", init="random"
)
data_2d_cos = model_cos.fit_transform(data_dists_fed)
data_2d_cos -= np.mean(data_2d_cos, axis=0)

fig, axs = plt.subplots(1, 2, figsize=(8,4.5))

random.seed(2)
data_2d = random.choices(
    list(zip(data_2d_fed, data_2d_cos, data)),
    k=args.plot_n
)

plot_data(
    axs[0], 0,
    [(x_fed, x) for x_fed, x_cos, x in data_2d],
    "Panphon FED"
)
plot_data(
    axs[1], 1,
    [(x_cos, x) for x_fed, x_cos, x in data_2d],
    "Metric learning"
)

plt.subplots_adjust(wspace=0)
plt.tight_layout(pad=1)
plt.show()
