#!/usr/bin/env python3

import argparse
import pickle
from main.utils import load_embd_data

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", nargs="+")
args.add_argument("-o", "--output")
args = args.parse_args()

data = []
for fname in args.input:
    data_local = load_embd_data(fname)
    print(fname, len(data_local))
    data += data_local

with open(args.output, "wb") as f:
    pickle.dump(data, f)