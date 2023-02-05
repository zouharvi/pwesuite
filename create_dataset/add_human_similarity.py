#!/usr/bin/env python3

import collections
import csv
from main.utils import load_multi_data, LANGS, UNK_SYMBOL
import panphon
import epitran

ft = panphon.FeatureTable()

if __name__ == '__main__':
    data = load_multi_data(path="data/multi_1.tsv", purpose_key="all")
    data_langs = collections.defaultdict(list)
    for line in data:
        data_langs[line[2]].append(line)
    print("prev en", len(data_langs["en"]))

    epi = epitran.Epitran("eng-Latn")
    vocab_ipa_multi = set(open("data/vocab/ipa_multi.txt").read().split())

    with open("data/human_similarity.csv", "r") as f:
        data_hs = list(csv.DictReader(f))
        data_hs_words = list(
            set(x["word1"] for x in data_hs) | set(x["word2"] for x in data_hs)
        )
        for word in data_hs_words:
            segments = ft.ipa_segs(epi.transliterate(word))
            segments = [s if s in vocab_ipa_multi else UNK_SYMBOL for s in segments]

            data_langs["en"].append((
                word,
                ''.join(segments),
                "en",
                "human_similarity",
                "",
            ))
        print("now en", len(data_langs["en"]))

    with open("data/multi.tsv", "w") as f:
        for lang in LANGS:
            for line in data_langs[lang]:
                f.write("\t".join(line) + "\n")
