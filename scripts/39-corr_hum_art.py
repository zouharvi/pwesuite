#!/usr/bin/env python3

import csv
import collections
import epitran
import numpy as np
import panphon2
from scipy.stats import pearsonr, spearmanr

transliterate = epitran.Epitran("eng-Latn").transliterate
fed = panphon2.FeatureTable().feature_edit_distance

def compute_panphon_distance(y, data):
    return [fed(w, y) for w, _ in data]

with open("data/human_similarity.csv", "r") as f:
    data_hs = list(csv.DictReader(f))



batches = collections.defaultdict(list)
for w_hs in data_hs:
    word2_ipa = transliterate(w_hs["word2"])
    batches[w_hs["word2"]].append((
        fed(transliterate(w_hs["word1"]), word2_ipa),
        w_hs["obtained"],
    ))
corr_pearson_all = []
corr_spearman_all = []
for batch in batches.values():
    predicted = [float(p) for p, _ in batch]
    obtained = [float(o) for _, o in batch]

    corr_pearson_all.append(pearsonr(predicted, obtained)[0])
    corr_spearman_all.append(spearmanr(predicted, obtained)[0])

print(f"Pearson: {np.average(corr_pearson_all):.2%}")
print(f"Spearman: {np.average(corr_spearman_all):.2%}")