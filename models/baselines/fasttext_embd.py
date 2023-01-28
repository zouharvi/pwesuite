#!/usr/bin/env python3

import argparse
import os
from main.utils import load_multi_data
import pickle
import numpy as np
import tqdm
import fasttext.util
import fasttext


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-o", "--output", default="computed/embd_fasttext.pkl")
args = args.parse_args()

data = load_multi_data(args.data)

if not os.path.exists("computed/fasttext"):
    os.makedirs("computed/fasttext")

def get_model(lang):
    model_path = f'cc.{lang}.300.bin'
    if not os.path.isfile("computed/fasttext/" + model_path):
        fasttext.util.download_model(lang, if_exists='ignore')

    if os.path.isfile(model_path):
        os.rename(model_path, "computed/fasttext/"+model_path)

    model = fasttext.load_model("computed/fasttext/" + model_path)
    return model

loaded_lang = None

data_out = []

for (word, _, lang, _) in tqdm.tqdm(data):
    if loaded_lang != lang:
        loaded_lang = lang
        model = get_model(lang) 

    vector = model.get_word_vector(word)
    # print(vector.shape, type(vector))
    data_out.append(vector)

# make sure dimensions fit
assert all(
    l is None or len(l) == 300
    for l in data_out
)
assert len(data_out) == len(data)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)