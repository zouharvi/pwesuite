#!/usr/bin/env python3

import argparse
from main.utils import load_multi_data
import pickle
from bpemb import BPEmb
import tqdm

args = argparse.ArgumentParser()
args.add_argument("-o", "--output", default="computed/embd_baseline/bpemb.pkl")
args = args.parse_args()

data = load_multi_data(purpose_key="all")

def get_model(lang):
    # use the English model for multi language
    if lang == "multi":
        lang = "en"
    # the loader will fall back to largest available
    return BPEmb(lang=lang, dim=300, vs=200000)

loaded_lang = None

data_out = []

for x in tqdm.tqdm(data):
    if loaded_lang != x["lang"]:
        loaded_lang = x["lang"]
        model = get_model(x["lang"]) 

    vector = model.embed(x["token_ort"]).mean(axis=0)
    data_out.append(vector)

# make sure dimensions fit
assert all(
    l is None or len(l) == 300
    for l in data_out
)
assert len(data_out) == len(data)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)