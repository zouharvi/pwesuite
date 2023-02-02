#!/usr/bin/env python3
import json

import numpy as np
import main.fig_utils
import matplotlib.pyplot as plt
import glob
import json
from scipy.stats import pearsonr, spearmanr

data = []
for fname in glob.glob("logs/eval_all_rnn_panphon_s*.log"):
    dim = int(fname.split("/")[-1].removesuffix(".log").split("_s")[-1])
    text = open(fname, "r").readlines()
    text = [l for l in text if l.startswith("JSON1!")][0]
    text = text[len("JSON1!"):]
    data.append((dim, json.loads(text)))

plt.figure(figsize=(3.5,2.2))
ax = plt.gca()

ax.set_xscale('log')

# sort by train sizes
data.sort(key=lambda x: x[0])
task_keys = list(data[0][1].keys())

for task in task_keys:
    plt.plot(
        [x[0]*1000 for x in data],
        [x[1][task] for x in data],
        label=task.title(),
        linewidth=2.5 if task == "overall" else 2,
        color="black" if task == "overall" else None,
        linestyle="--" if task == "overall" else "-",
        
    )

corr_spearman = []
corr_pearson = []
for task1_i, task1 in enumerate(task_keys):
    task1vals = [x[1][task1] for x in data]
    for task2 in task_keys[task1_i+1:]:
        task2vals = [x[1][task2] for x in data]
        corr_spearman.append(spearmanr(task1vals, task2vals)[0])
        corr_pearson.append(pearsonr(task1vals, task2vals)[0])

print(f"Avg. Spearman {np.average(corr_spearman)}")
print(f"Avg. Pearson {np.average(corr_pearson)}")

plt.legend(
    ncol=3,
    bbox_to_anchor=(-0.1, 1.05, 1.1, 0), loc="lower left",
    mode="expand",
)


plt.ylabel("Score")
plt.xlabel("Training data size (logarithmic scale)")

plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/sizes_perf.pdf")
plt.show()