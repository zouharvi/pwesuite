#!/usr/bin/env python3
import json

import numpy as np
import main.fig_utils
import matplotlib.pyplot as plt
import glob
import json
import collections
from scipy.stats import pearsonr, spearmanr
import scipy.stats as st

def get_interval(data):
    mean = np.mean(data)
    interval = st.t.interval(
        confidence=0.90, df=len(data)-1,
        loc=mean,
        scale=st.sem(data)
    )
    return (interval[0], mean, interval[1])


data = collections.defaultdict(list)
for fname in glob.glob("logs/eval_all_rnn_panphon_d*.log"):
    dim = fname.split("/")[-1].removesuffix(".log").split("_d")[-1].split("_")
    if len(dim) != 2:
        continue
    dim = int(dim[0])

    text = open(fname, "r").readlines()
    text = [l for l in text if l.startswith("JSON1!")]
    if not text:
        continue
    text = text[0][len("JSON1!"):]
    data[dim].append(json.loads(text))

DIMS = list(set(data.keys()))
DIMS.sort()

plt.figure(figsize=(3.5, 2.2))
ax = plt.gca()

# sort by dims
task_keys = list(data[DIMS[0]][0].keys())

TASK_NAMES = {
    "human_similarity": "Human Sim.",
    "correlation": "Art. Dist."
}
def get_task_name(task):
    if task in TASK_NAMES:
        return TASK_NAMES[task]
    if task == "overall":
        return None
    return task.title()

for task_i, task in enumerate(task_keys):
    values = [get_interval([x[task] for x in data[d]]) for d in DIMS]
    ax.fill_between(
        DIMS,
        [x[0] for x in values],
        [x[1] for x in values],
        color="black" if task == "overall" else main.fig_utils.COLORS[task_i],
        alpha=0.3,
    )
    plt.plot(
        DIMS,
        [x[1] for x in values],
        label=get_task_name(task),
        linewidth=2.5 if task == "overall" else 2,
        color="black" if task == "overall" else main.fig_utils.COLORS[task_i],
        linestyle="--" if task == "overall" else "-",
    )

corr_spearman = []
corr_pearson = []
for task1_i, task1 in enumerate(task_keys):
    task1vals = [np.mean([x[task1] for x in data[d]]) for d in DIMS]
    # task1vals = [x[1][task1] for x in data]
    for task2 in task_keys[task1_i + 1:]:
        task2vals = [np.mean([x[task2] for x in data[d]]) for d in DIMS]
        corr_spearman.append(spearmanr(task1vals, task2vals)[0])
        corr_pearson.append(pearsonr(task1vals, task2vals)[0])

print(f"Avg. Spearman {np.average(corr_spearman)}")
print(f"Avg. Pearson {np.average(corr_pearson)}")

plt.legend(
    ncol=3,
    bbox_to_anchor=(-0.13, 1.05, 1.15, 0), loc="lower left",
    mode="expand",
)


plt.ylabel("Score")
plt.xlabel("Dimensions")

plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/dimensions_perf.pdf")
plt.show()
