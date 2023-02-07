#!/usr/bin/env python3
import json
import main.fig_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from main.utils import LANGS

data = open("computed/mismatch_tokenort.log", "r").readlines()
data1 = [json.loads(l[len("JSON1!"):]) for l in data if l.startswith("JSON1!")][0]
data2 = [json.loads(l[len("JSON2!"):]) for l in data if l.startswith("JSON2!")][0]

for task_key in data2["en-en"].keys():
    scores = [v[task_key] for  k, v in data2.items()]
    print(f"{task_key}: {np.average(scores):.2f}")


img = np.zeros((len(LANGS), len(LANGS)))

plt.figure(figsize=(3.5,2.5))
ax = plt.gca()

for langs, score in data1.items():
    lang1, lang2 = langs.split("-")
    lang1_i = LANGS.index(lang1)
    lang2_i = LANGS.index(lang2)
    
    # print(lang1, lang2, lang1_i, lang2_i, score)
    # the intensity shows how much better it is than itself (in column)
    val = score/data1[lang1+"-"+lang1]
    img[lang2_i, lang1_i] = val
    plt.text(
        lang1_i, lang2_i, f"{score:.2f}",
        va="center", ha="center",
        color="white" if val >= 0.99 else "black"
    )

    if lang1_i == lang2_i:
        ax.add_patch(
            Rectangle(
                (lang1_i-0.5, lang2_i-0.5), 1, 1,
                fill=False,
                edgecolor="black", linewidth=1.5,
                clip_on=False,
            )
        )


# reverse values
plt.imshow(-img, aspect="auto", cmap="summer")
plt.xticks(range(len(LANGS)), [x.upper() for x in LANGS])
plt.yticks(range(len(LANGS)), [x.upper() for x in LANGS])
plt.ylabel("Train language")
plt.xlabel("Eval language")

plt.tight_layout(pad=0.2)
plt.savefig("computed/figures/language_mismatch.pdf")
plt.show()