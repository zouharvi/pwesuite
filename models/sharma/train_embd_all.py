#!/usr/bin/env python3

import torch
import argparse
import pickle
import tqdm
from main.utils import get_device
from wsim.wsim import wsimdict as WSimDict
from itertools import islice
import multiprocess as mp

DEVICE = get_device()
# DEVICE = "cpu"

args = argparse.ArgumentParser()
args.add_argument("-l", "--lang", default="en")
args.add_argument("-d", "--dimensions", type=int, default=300)
args.add_argument("--ntasks", type=int, default=20)
args.add_argument("--batch-size", type=int, default=5000)
args = args.parse_args()

def get_similarity_fast(data_batch):
    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())


    def compute_similarity_sub(data_local):
        wd = WSimDict(f'data/tmp/cmu_{args.lang}.txt')
        sim_nums = []
        for word in data_local:
            sim_local_nums = [
                wd.similarity(word, x, wd.BIGRAM | wd.INSERT_BEG_END | wd.VOWEL_BUFF, 1) for x in data_batch
            ]
            sim_nums.append(sim_local_nums)

        return sim_nums

    with mp.Pool(args.ntasks) as pool:
        sim_nums_chunked = pool.map(
            compute_similarity_sub, chunk(data_batch, args.batch_size//args.ntasks)
        )
    sim_nums = [x for l in sim_nums_chunked for x in l]
    
    return sim_nums

with open(f'data/tmp/cmu_{args.lang}.txt', "r") as f:
    data = [x.upper().rstrip("\n").split("  ") for x in f.readlines()]

embds = torch.randn((len(data), args.dimensions), requires_grad=True, device=DEVICE)

optimizer = torch.optim.Adam([embds], lr=5e-2)
loss_fn = torch.nn.MSELoss()
loss_last = 99999

for epoch in range(10000):
    random_batch = torch.randint(0, len(data)-1, (args.batch_size,))

    data_sims = get_similarity_fast([data[i][0] for i in random_batch])
    data_sims = torch.tensor(data_sims, requires_grad=False, device=DEVICE)
    embds_local = embds[random_batch,:]
    projection = torch.matmul(embds_local, embds_local.T)

    loss = loss_fn(projection, data_sims)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.detach().cpu().numpy()
    print("loss", loss)

    if epoch % 10 == 0:
        print(f"LOG!maybe saving at epoch {epoch} with loss {loss}")
        if loss < loss_last:
            print(f"LOG!saving at epoch {epoch} with loss {loss}")
            loss_last = loss
            with open(f"computed/tmp/sharma_embd_{args.lang}.pkl", "wb") as f:
                pickle.dump(embds.cpu().tolist(), f)
        else:
            print(f"LOG!not saving at epoch {epoch} with loss {loss} because last loss is {loss_last}")