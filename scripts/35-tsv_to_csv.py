#!/usr/bin/env python3

raise Exception("Deprecated")
import csv

with open("data/multi_0.tsv", "r") as f:
    data = list(csv.reader(f, delimiter="\t"))

print("loaded", len(data), "words")

FIELDNAMES_OLD = ["token_ort", "token_ipa", "lang", "purpose", "token_arp"]
FIELDNAMES_NEW = ["token_ort", "token_ipa", "token_arp", "lang", "purpose"]
print("fieldnames old:", FIELDNAMES_OLD)
print("fieldnames new:", FIELDNAMES_NEW)

with open("data/multi.csv", "w") as f:
    w = csv.DictWriter(
        f,
        fieldnames=FIELDNAMES_NEW,
        quoting=csv.QUOTE_ALL,
    )
    w.writeheader()
    # there's probably a way to do this automatically
    for line in data:
        w.writerow({
            "token_ort": line[0],
            "token_ipa": line[1],
            "token_arp": line[4],
            "lang": line[2],
            "purpose": line[3]
        })