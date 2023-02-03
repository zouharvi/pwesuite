#!/usr/bin/env python3

import argparse
import os
from main.utils import load_multi_data
import pickle
import tqdm
import fasttext.util
import fasttext


args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-o", "--output", default="computed/embd_baseline/fasttext.pkl")
args = args.parse_args()

data = load_multi_data(args.data, purpose_key="all")

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

for x in tqdm.tqdm(data):
    if loaded_lang != x[2]:
        loaded_lang = x[2]
        model = get_model(x[2]) 

    vector = model.get_word_vector(x[0])
    data_out.append(vector)

# make sure dimensions fit
assert all(
    l is None or len(l) == 300
    for l in data_out
)
assert len(data_out) == len(data)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)