#!/usr/bin/env python3

import collections
import csv
import os
import pickle

import tqdm
from main.utils import load_multi_data, LANGS, UNK_SYMBOL
import panphon
import epitran
from main.ipa2arp import IPA2ARP
from collections import Counter
from Levenshtein import ratio

ipa2arp = IPA2ARP().convert
ft = panphon.FeatureTable()
lang2epi = {
    "eng": epitran.Epitran("eng-Latn"),
    'amh': epitran.Epitran('amh-Ethi'),
    'ben': epitran.Epitran('ben-Beng'),
    'uzb': epitran.Epitran('uzb-Latn'),
    'pol': epitran.Epitran('pol-Latn'),
    'spa': epitran.Epitran('spa-Latn'),
    'swa': epitran.Epitran('swa-Latn'),
    'fra': epitran.Epitran('fra-Latn'),
    'deu': epitran.Epitran('deu-Latn'),
}

lang2lang = {
    "eng": "en",
    "deu": "de",
    "amh": "am",
    "fra": "fr",
    "ben": "bn",
    "uzb": "uz",
    "pol": "pl",
    "spa": "es",
    "swa": "sw",
}
OUR_LANGS = {"eng", "amh", "ben", "uzb", "pol", "spa", "swa", "fra", "deu"}

def get_closest_distractor(word, ban_concept, data_cognates_flat):
    def modified_lev(distractor):
        if distractor[0] == ban_concept:
            return 0
        val = ratio(word, distractor[2])
        if val > 0.8:
            return 0
        return val

    data_cognates_flat = [x for x in data_cognates_flat if x [0] != ban_concept]
    distractor = max(data_cognates_flat, key=modified_lev)
    return distractor

def get_cognates():
    CACHE_PATH = f"data/cache/cognates.pkl"
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    with open("data/raw/CogNet-v0.tsv", "r") as f:
        data_cognates_raw = [
            line.rstrip("\n").split("\t")
            for line in f.readlines()
        ]
    
    data_cognates_flat = [
        y for x in data_cognates_raw
        for y in [(x[0], *x[1:3]), (x[0], *x[3:5])]
        if y[1] in OUR_LANGS
    ]

    # filter to supported langs
    data_cognates_raw = [x for x in data_cognates_raw if x[1] in OUR_LANGS and x[3] in OUR_LANGS]

    data_cognates = []
    for concept, lang1, word1, lang2, word2, translit1, translit2 in tqdm.tqdm(data_cognates_raw):
        if word1 == word2:
            continue
        distractor = get_closest_distractor(word1, concept, data_cognates_flat)
        data_cognates.append((
            {
                "lang": lang1,
                "word": word1,
            }, {
                "lang": lang2,
                "word": word2,
            },
            {
                "lang": distractor[1],
                "word": distractor[2],
            }
        ))

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(data_cognates, f)

    print("Got", len(data_cognates), "cognates")

    return data_cognates

if __name__ == '__main__':
    data_cognates = get_cognates()

    data = load_multi_data(path="data/multi_2.tsv", purpose_key="all")
    data_langs = collections.defaultdict(list)
    for line in data:
        data_langs[line[2]].append(line)
    
    # create a set of existing words to avoid duplicities
    data_lang_words = collections.defaultdict(set)
    for lang in LANGS:
        for line in data_langs[lang]:
            data_lang_words[lang].add(line[0])

    vocab_ipa_multi = set(open("data/vocab/ipa_multi.txt").read().split())

    for word1, word_pos, word_neg in data_cognates:
        for word in [word1, word_pos, word_neg]:
            # update duplicates check
            if word["word"] in data_lang_words[word["lang"]]:
                continue
            data_lang_words[word["lang"]].add(word["word"])

            epi = lang2epi[word["lang"]]
            segments = ft.ipa_segs(epi.transliterate(word["word"]))
            segments = [s if s in vocab_ipa_multi else UNK_SYMBOL for s in segments]
            token_ipa = ''.join(segments)

            data_langs[lang2lang[word["lang"]]].append((
                word["word"],
                token_ipa,
                lang2lang[word["lang"]],
                "cognates",
                " ".join(ipa2arp(token_ipa)),
            ))

    with open("data/multi.tsv", "w") as f:
        for lang in LANGS:
            for line in data_langs[lang]:
                f.write("\t".join(line) + "\n")
