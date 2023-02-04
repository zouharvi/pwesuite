#!/usr/bin/env python3

import csv

with open("data/multi.tsv", "r") as f:
    data = list(csv.reader(f, delimiter="\t"))

print("loaded", len(data), "words")

FIELDNAMES = ["token_ort", "token_ipa", "lang", "purpose", "token_arp"]
print("fieldnames:", FIELDNAMES)

with open("data/multi.csv", "w") as f:
    w = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES,
        quoting=csv.QUOTE_ALL,
    )
    w.writeheader()
    for line in data:
        w.writerow({k:v for k,v in zip(FIELDNAMES, line)})

# TODO create all, train, dev