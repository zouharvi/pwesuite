#!/usr/bin/env python3

import argparse
import math
from InstructorEmbedding import INSTRUCTOR
from main.utils import load_multi_data, get_device
import pickle
import numpy as np
import tqdm

BATCH_SIZE=500

args = argparse.ArgumentParser()
args.add_argument("-d", "--data", default="data/multi.tsv")
args.add_argument("-o", "--output", default="computed/embd_baseline/instructor.pkl")
args = args.parse_args()

data = load_multi_data(args.data)
data = [x[0] for x in data]

model = INSTRUCTOR('hkunlp/instructor-large').to(get_device())
instruction = "Represent the word for sound similarity retrieval:"

data_out = []


for i in tqdm.tqdm(range(math.ceil(len(data)/BATCH_SIZE))):
    embeddings = list(model.encode([
        [instruction,word]
        for word in data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    ]))

    data_out += embeddings

# make sure dimensions fit
assert all(len(l) == 768 for l in data_out)
assert len(data_out) == len(data)

with open(args.output, "wb") as f:
    pickle.dump(data_out, f)