#!/usr/bin/env python3

import argparse
import math
from transformers import pipeline
from main.utils import load_multi_data, get_device
import pickle
import numpy as np
import tqdm

BATCH_SIZE=5000

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-o", "--output", default="computed/embd_bert.pkl")
args = args.parse_args()

data = load_multi_data(args.data)
data = [x[0] for x in data]

model = pipeline(
    'feature-extraction',
    model='bert-base-multilingual-uncased',
    device=get_device()
)

data_out = []

for i in tqdm.tqdm(range(math.ceil(len(data)/BATCH_SIZE))):
    model_output = model(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

    # make sure the formatting fits and we don't lose any information
    assert all(len(l) == 1 for l in model_output)

    # do averaging
    data_out += [np.array(l[0]).mean(axis=0) for l in model_output]

# make sure dimensions fit
assert all(len(l) == 768 for l in data_out)
assert len(data_out) == len(data)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)