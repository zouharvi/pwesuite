#!/usr/bin/env python3

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from main.utils import load_multi_data, LANGS
import argparse
import panphon

args = argparse.ArgumentParser()
args.add_argument("--features", default="tokenort")
args.add_argument("--vectorizer", default="tfidf")
args.add_argument("--nopca", action="store_true")
args.add_argument("--norm", action="store_true")
args = args.parse_args()

data = load_multi_data(purpose_key="all")
ft = panphon.FeatureTable()


def process_one_lang(lang):
    if args.features == "tokenort":
        data_local = [" ".join(x[0]) for x in data if x[2] == lang]
    elif args.features == "tokenipa":
        data_local = [
            " ".join(ft.ipa_segs(x[1]))
            for x in data if x[2] == lang
        ]

    vectorizer_args = {
        "max_features": 300 if args.nopca else 1024,
        "ngram_range": (1, 3),
        "stop_words": None,
        "analyzer": "char",
        "min_df": 0,
    }

    if args.vectorizer == "tfidf":
        vectorizer = TfidfVectorizer(**vectorizer_args)
    elif args.vectorizer == "count":
        vectorizer = CountVectorizer(**vectorizer_args)

    print("Transforming")
    data_local = vectorizer.fit_transform(data_local)
    data_local = np.asarray(data_local.todense())
    print(data_local.shape)

    if not args.nopca:
        print("Performing PCA")
        pca = PCA(n_components=300, whiten=True)
        data_local = pca.fit_transform(data_local)

    if args.norm:
        data_local /= np.linalg.norm(data_local, axis=1, ord=1)[:, None]

    return list(data_local)


data_out = []
for lang in LANGS + ["multi"]:
    print(lang)
    data_out += process_one_lang(lang)

with open(f"computed/embd_other/count_{args.features}_{args.vectorizer}{'_nopca' if args.nopca else ''}{'_norm' if args.norm else ''}.pkl", "wb") as f:
    pickle.dump(data_out, f)
