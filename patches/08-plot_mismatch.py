#!/usr/bin/env python3
import json
import main.fig_utils
import numpy as np
import matplotlib.pyplot as plt

data = open("computed/mismatch_tokenort.log", "r").readlines()
data1 = [json.loads(l[len("JSON1!"):]) for l in data if l.startswith("JSON1!")][0]
data2 = [json.loads(l[len("JSON2!"):]) for l in data if l.startswith("JSON2!")][0]

for task_key in data2["en-en"].keys():
    scores = [v[task_key] for  k, v in data2.items()]
    print(f"{task_key}: {np.average(scores):.2f}")

langs_all = ['en', 'am', 'bn', 'uz', 'pl', 'es', 'sw']

img = np.zeros((len(langs_all), len(langs_all)))

plt.figure(figsize=(4,3))

for langs, score in data1.items():
    lang1, lang2 = langs.split("-")
    lang1_i = langs_all.index(lang1)
    lang2_i = langs_all.index(lang2)
    
    # print(lang1, lang2, lang1_i, lang2_i, score)
    # the intensity shows how much better it is than itself (in column)
    val = score/data1[lang1+"-"+lang1]
    img[lang2_i, lang1_i] = val
    plt.text(
        lang1_i, lang2_i, f"{score:.2f}",
        va="center", ha="center",
        color="white" if val <= 0.95 else "black"
    )


plt.imshow(img, aspect="auto")
plt.xticks(range(len(langs_all)), [x.upper() for x in langs_all])
plt.yticks(range(len(langs_all)), [x.upper() for x in langs_all])

plt.tight_layout()
plt.show()