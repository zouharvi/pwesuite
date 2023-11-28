#!/usr/bin/env python3

import os
import datasets
import csv

os.makedirs("data/", exist_ok=True)

dataset = datasets.load_dataset("zouharvi/pwesuite-eval")
writer = csv.writer(open("data/multi.tsv", "w"), delimiter="\t")

for line in dataset["train"]:
    writer.writerow([
        line["token_ort"],
        line["token_ipa"],
        line["lang"],
        line["purpose"],
        line["token_arp"],
    ])