#!/usr/bin/env python3 

from collections import Counter
import tqdm
import numpy as np
import pickle
from sklearn.decomposition import PCA
from main.utils import load_multi_data, LANGS
from featurephone import feature_bigrams
from main.ipa2cmu import IPA2CMU

ipa2cmu = IPA2CMU().convert

def normalize(vec):
    """Return unit vector for parameter vec.

    >>> normalize(np.array([3, 4]))
    array([ 0.6,  0.8])

    """
    if np.any(vec):
        norm = np.linalg.norm(vec)
        return vec / norm
    else:
        return vec

data = load_multi_data(purpose_key="all")

def process_one_lang(lang):
    all_features = Counter()
    entries = list()
    data_local = [x for x in data if x[2] == lang]

    for line in tqdm.tqdm(data_local):
        word = line[0]
        phones = line[4].split()
        if not phones:
            # fall back to automatic conversion for words that are not in CMU arpabet
            # this is true for some English and all non-English words
            phones = ipa2cmu(line[1])
        features = Counter(feature_bigrams(phones))
        entries.append((word, features))
        all_features.update(features.keys())

    print("Entries:", len(entries))

    filtfeatures = sorted([
        f for f, count in all_features.items()
        if count >= 2
    ])

    print("Feature count:", len(filtfeatures))

    arr = np.array([
        normalize([i.get(j, 0) for j in filtfeatures])
        for word, i in entries
    ])

    print("Performing PCA")
    pca = PCA(n_components=300, whiten=True)
    transformed = pca.fit_transform(arr)
    return list(transformed)


data_out = []
for lang in LANGS:
    print(lang)
    data_out += process_one_lang(lang)

with open("computed/embd_other/parrish_pca.pkl", "wb") as f:
    pickle.dump(data_out, f)
