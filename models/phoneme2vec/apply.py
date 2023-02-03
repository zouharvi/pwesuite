#!/usr/bin/env python3

import pickle
import numpy as np
import tqdm
from main.ipa2cmu import IPA2CMU
from main.utils import load_multi_data

ipa2cmu = IPA2CMU().convert

with open("computed/phoneme2vec", "rb") as f:
    phonemes, vecs = pickle.load(f)

assert len(phonemes) == len(vecs)

mapping = {p:v for p, v in zip(phonemes,vecs)}

def get_embd(arp):
    if arp in mapping:
        return [mapping[arp]]
    
    out = []
    while True:
        if len(arp) == 0:
            break
        found = False
        for end_i in range(len(arp)-1, 0, -1):
            if arp[:end_i] in mapping:
                out.append(mapping[arp[:end_i]])
                arp = arp[:end_i]
                found = True
                break
        if not found:
            arp = arp[1:]
    return out



data = load_multi_data(purpose_key="all")
data_out = []

# TODO: multiprocess?
for line in tqdm.tqdm(data):
    phones = "".join([c for c in line[4].lower() if c.isalpha() or c==" "]).split()
    if not phones:
        # fall back to automatic conversion for words that are not in CMU arpabet
        # this is true for some English and all non-English words
        phones = ipa2cmu(line[1])

    embd = [get_embd(arp) for arp in phones]
    embd = [x for l in embd for x in l]
    if len(embd) == 0:
        embd = np.random.random(50)
    else:
        embd = np.array(embd)
        embd = embd.mean(axis=0)
    data_out.append(embd)

with open("computed/embd_other/phoneme2vec.pkl", "wb") as f:
    pickle.dump(data_out, f)
