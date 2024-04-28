#!/usr/bin/env python3

raise Exception("Deprecated")

import argparse
from main.utils import load_multi_data
import pickle
import spacy
import numpy as np
import tqdm


args = argparse.ArgumentParser()
args.add_argument("-o", "--output", default="computed/embd_baseline/spacy.pkl")
args = args.parse_args()

data = load_multi_data(purpose_key="all")

def get_model(lang):
    if lang in {"en"}:
        return spacy.load("en_core_web_lg")
    elif lang in {"pl"}:
        return spacy.load("pl_core_news_lg")
    elif lang in {"es"}:
        return spacy.load("es_core_news_lg")
    # 'am' 'bn' 'uz', 'sw'

loaded_lang = None

data_out = []

for (word, _, lang, _) in tqdm.tqdm(data):
    if loaded_lang != lang:
        loaded_lang = lang
        model = get_model(lang) 

    # spacy is unable to produce embeddings for some languages
    if model is None:
        data_out.append(None)
        continue
    
    vector = model(word).vector
    data_out.append(vector)

# make sure dimensions fit
assert all(
    l is None or len(l) == 300
    for l in data_out
)
assert len(data_out) == len(data)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)