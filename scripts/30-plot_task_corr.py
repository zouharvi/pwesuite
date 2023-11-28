#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.scripts import Polygon
from scipy.stats import pearsonr, spearmanr
import csv

with open("meta/evaluation_all.csv", "r") as f:
    data = list(csv.DictReader(f))

tasks = ["Human Sim.", "Art. Dist.","Retrieval", "Analogies", "Rhyme", "Cognate"]
data = [
    {task:float(line[task]) for task in tasks}
    for line in data
]

plt.figure(figsize=(3.5, 2.1))
ax = plt.gca()
# remove frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
img = np.full((len(tasks), len(tasks)), np.nan)

cmap = matplotlib.cm.Greys.copy().reversed()

vmin = -0.1
vmax = 1


for task1_i, task1 in enumerate(tasks):
    for task2_i, task2 in enumerate(tasks[task1_i+1:]):
        task1vals = [line[task1] for line in data]
        task2vals = [line[task2] for line in data]
        corr_s = spearmanr(task1vals, task2vals)[0]
        corr_p = pearsonr(task1vals, task2vals)[0]
        img[task2_i+task1_i+1][task1_i] = abs(corr_s)

        plt.text(
            x=task1_i-0.20,
            y=task2_i+task1_i-0.2,
            s=f"{corr_s:.2f}",
            va="center", ha="center",
            color="white" if corr_s <= 0.2 else "black",
        )

        plt.text(
            x=task1_i+0.2,
            y=task2_i+task1_i+0.2,
            s=f"{corr_p:.2f}",
            va="center", ha="center",
            color="white" if corr_p <= 0.2 else "black",
        )
        
        polygon_coords = np.array([
                [task1_i,task2_i+task1_i+1],
                [task1_i+1,task2_i+task1_i],
                [task1_i+1,task2_i+task1_i+1],
            ])
        ax.add_patch(Polygon(
            polygon_coords-0.5,
            color=cmap((corr_p-vmin)/(vmax-vmin)),
            linewidth=0,
        ))

img = np.ma.masked_invalid(img)
# trim image
img = img[1:,:-1]
# reverse values for nicer map
plt.imshow(
    img, aspect="auto", cmap=cmap,
    vmin=vmin, vmax=vmax,
)
plt.xticks(
    range(len(tasks)-1),
    [
        f"\n{t}" if t_i % 2 == 0 else t
        for t_i, t in enumerate(tasks[:-1])
    ]
)
plt.yticks(range(len(tasks)-1), tasks[1:])
plt.tight_layout()
plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/task_correlation.pdf")
plt.show()