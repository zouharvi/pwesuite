#!/usr/bin/env python3

import collections
import csv
from main.utils import load_multi_data_raw, LANGS, UNK_SYMBOL
import panphon
import epitran
from main.ipa2arp import IPA2ARP

ipa2arp = IPA2ARP().convert

ft = panphon.FeatureTable()

if __name__ == '__main__':
    data = load_multi_data_raw(path="data/multi_1.tsv", purpose_key="all")
    data_langs = collections.defaultdict(list)
    for line in data:
        data_langs[line[2]].append(line)
    print("prev en", len(data_langs["en"]))

    epi = epitran.Epitran("eng-Latn")
    vocab_ipa = set(open("data/vocab/ipa_multi.txt").read().split())

    with open("data/human_similarity.csv", "r") as f:
        data_hs = list(csv.DictReader(f))
        data_hs_words = list(
            set(x["word1"] for x in data_hs) | set(x["word2"] for x in data_hs)
        )
        for word in data_hs_words:
            token_ipa = ft.ipa_segs(epi.transliterate(word))

            data_langs["en"].append((
                # assume all characters are in the vocabulary for this dataset
                word,
                "".join(token_ipa),
                "en",
                "human_similarity",
                " ".join(ipa2arp("".join(token_ipa))),
            ))
        print("now en", len(data_langs["en"]))

    with open("data/multi_2.tsv", "w") as f:
        for lang in LANGS:
            for line in data_langs[lang]:
                f.write("\t".join(line) + "\n")
