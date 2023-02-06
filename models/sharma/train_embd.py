#!/usr/bin/env python3

import torch
import argparse
import pickle
from main.utils import get_device

# DEVICE = get_device()
DEVICE = "cpu"

args = argparse.ArgumentParser()
args.add_argument("-l", "--lang", default="en")
args.add_argument("-d", "--dimensions", type=int, default=300)
args = args.parse_args()

# with open(f"computed/tmp/sharma_sim_{args.lang}.pkl", "rb") as f:
#     data = pickle.load(f)

data = torch.rand((20*1000, 20*1000))

data_sims = torch.tensor(data, requires_grad=False, device=DEVICE)
embds = torch.randn((len(data), args.dimensions), requires_grad=True, device=DEVICE)

optimizer = torch.optim.Adam([embds], lr=1e-2)
loss_fn = torch.nn.MSELoss()

for epoch in range(1000):
    projection = torch.matmul(embds, embds.T)
    loss = loss_fn(projection, data_sims)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss", loss)

with open(f"computed/tmp/sharma_embd_{args.lang}.pkl", "wb") as f:
    pickle.dump(embds.cpu().tolist(), f)