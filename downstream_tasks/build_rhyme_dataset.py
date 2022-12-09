#!/usr/bin/env python3

import numpy as np
import pickle
import argparse
import collections
import tqdm
import re

args = argparse.ArgumentParser()
# make sure it's re-saved with UTF-8, otherwise it will cause issues
args.add_argument("-i", "--input", default="data/raw/cmudict-0.7b")
args.add_argument("-o", "--output", default="data/rhymes.pkl")
args = args.parse_args()

with open(args.input, "r") as f:
    data = [x.rstrip("\n").split("  ") for x in f.readlines() if x[0].isalpha()]

print(f"Loaded {len(data)//1000}k words")

RE_LAST_STRESSED = re.compile(r".* ([\w]+1.*)")
rhyme_clusters = collections.defaultdict(list)
for word, pronunc in tqdm.tqdm(data):
    if "1" not in pronunc:
        continue
    rhyme_part = RE_LAST_STRESSED.match(" " + pronunc).group(1)
    rhyme_clusters[rhyme_part].append(word.lower())

print(len(rhyme_clusters), "rhyme clusters with")
print(f"average cluster size: {np.average([len(x) for x in rhyme_clusters.values()]):.1f}")

with open(args.output, "wb") as f:
    pickle.dump(rhyme_clusters, f)