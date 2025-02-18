#!/usr/bin/env python3

import tqdm
import pickle
import argparse
from wsim.wsim import wsimdict as WSimDict
from itertools import islice
from multiprocessing.pool import ThreadPool

args = argparse.ArgumentParser()
args.add_argument("-l", "--lang", default="en")
args.add_argument("--ntasks", type=int, default=5)
args = args.parse_args()

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

with open(f'data/tmp/cmu_{args.lang}.txt', "r") as f:
    data = [x.rstrip("\n").split("  ") for x in f.readlines()]

def compute_similarity_sub(data_local):
    wd = WSimDict(f'data/tmp/cmu_{args.lang}.txt')

    sim_nums = []
    for word in tqdm.tqdm(data_local):
        sim_local_nums = [
            wd.similarity(word[0], x[0], wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1) for x in data
        ]
        sim_nums.append(sim_local_nums)

    return sim_nums

data_chunked = list(chunk(data, 1000))

with ThreadPool(args.ntasks) as pool:
    sim_nums_chunked = pool.map(
        compute_similarity_sub, tqdm.tqdm(data_chunked)
    )

sim_nums = [x for l in sim_nums_chunked for x in l]
print(len(data), len(sim_nums_chunked), len(sim_nums))

with open(f"computed/tmp/sharma_{args.lang}.pkl", "wb") as f:
    pickle.dump(sim_nums, f)

# r = wd.top_similar('SIT', 10, wd.BIGRAM | wd.INSERT_BEG_END, 1)
# r = wd.similarity(data[10][0], data[200][0], wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1)
# r = wd.random_scores(5, wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1)
# r = [wd.get_index(s.upper()) for s in words]
