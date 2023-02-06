#!/usr/bin/env python3

import os
from main.utils import load_multi_data, LANGS, UNK_SYMBOL
import random
random.seed(0)

os.makedirs("data/tmp", exist_ok=True)

data = load_multi_data(purpose_key="all")

ALLOWED_ARP = {
    "P", "B", "F", "V", "TH", "DH", "N", "T", "D", "S", "Z", "R", "L", "SH", "ZH", "Y", "NG", "K", "G", "M",
    "W", "HH", "CH", "JH", "AO", "AA", "IY", "UW", "EH", "IH", "UH", "AH", "AE", "EY", "AY", "OW", "AW", "OY", "ER",
}
ALLOWED_ARP_l = list(ALLOWED_ARP)

UNK_TO_ALLOWED = {
    "H": "HH",
    "DX": "K",
    "IX": "Z",
    "Q": "G",
    "HV": "W",
    "AX": "S",
}


missed_random = 0

def normalize_arp(txt):
    global missed_random
    txt_orig = str(txt)
    txt = "".join([x for x in txt if x.isalpha() or x.isspace()])
    arps_orig = txt.split(" ")
    arps = [
        x if x in ALLOWED_ARP else UNK_TO_ALLOWED[x]
        for x in arps_orig if x in ALLOWED_ARP or x in UNK_TO_ALLOWED
    ]
    if len(arps) == 0:
        arps = random.choices(ALLOWED_ARP_l, k=len(arps_orig))
        if len(txt_orig.replace(" ", "").replace(UNK_SYMBOL, "")) > 0:
            print("random substitution", txt_orig, arps_orig, arps)
        missed_random += 1

    return " ".join(arps)

for lang in LANGS:
    print(lang)
    data_local = [
        x for x in data if x[2] == lang
    ]
    data_local = [
        x[0].upper() + "  " + normalize_arp(x[4])
        for x in data_local
    ]
    with open(f"data/tmp/cmu_{lang}.txt", "w") as f:
        f.write("\n".join(data_local))

# TODO: note in paper that only English is comparable
print("Total randomly substituted", missed_random)