#!/usr/bin/env python3

import os
import pickle
import tqdm
from main.utils import load_multi_data_raw, UNK_SYMBOL
import panphon
import epitran
from main.ipa2arp import IPA2ARP
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


def get_closest_distractor(word, word_trans, ban_concept, ban_lang, data_cognates_flat):
    if len(word_trans) == 0:
        word_trans = word

    def modified_lev(distractor):
        distractor_w = distractor[3]
        if len(distractor_w) == 0:
            distractor_w = distractor[2]

        if distractor[0] == ban_concept:
            return 0
        val = ratio(word_trans, distractor_w)
        if val > 0.75:
            return 0
        return val

    distractor = max(data_cognates_flat, key=modified_lev)
    return distractor

def generate_cognates():
    from urllib.request import urlretrieve
    os.makedirs("data/raw", exist_ok=True)
    urlretrieve("https://raw.githubusercontent.com/kbatsuren/CogNet/master/CogNet-v0.tsv", "data/raw/CogNet-v0.tsv")

    with open("data/raw/CogNet-v0.tsv", "r") as f:
        data_cognates_raw = [
            line.rstrip("\n").split("\t")
            for line in f.readlines()
        ]

    data_cognates_flat = [
        y for x in data_cognates_raw
        for y in [(x[0], *x[1:3], x[5]), (x[0], *x[3:5], x[6])]
        if y[1] in OUR_LANGS
    ]

    # filter to supported langs
    data_cognates_raw = [
        x for x in data_cognates_raw
        if x[1] in OUR_LANGS and x[3] in OUR_LANGS
    ]

    data_cognates = []
    for concept, lang1, word1, lang2, word2, translit1, translit2 in tqdm.tqdm(data_cognates_raw):
        if word1 == word2:
            continue
        distractor = get_closest_distractor(
            word1, translit1, concept,
            lang1, data_cognates_flat
        )
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

    print("Got", len(data_cognates), "cognates")

    return data_cognates


if __name__ == '__main__':
    data_cognates = generate_cognates()
    data = load_multi_data_raw(path="data/multi_2.tsv", purpose_key="all")

    vocab_ipa = set(open("data/vocab/ipa_multi.txt").read().split())
    vocab_ort = set(open("data/vocab/ort_multi.txt").read().split())

    seen_words = set()
    data_cog = []
    for cognate_i, (word1, word_pos, word_neg) in enumerate(data_cognates):
        if word1["word"] in seen_words:
            continue
        for word_i, word in enumerate([word1, word_pos, word_neg]):
            # update duplicates check
            seen_words.add(word["word"])

            epi = lang2epi[word["lang"]]
            token_ipa_raw = ft.ipa_segs(epi.transliterate(word["word"]))
            token_ipa = ''.join([
                s if s in vocab_ipa else UNK_SYMBOL
                for s in token_ipa_raw
            ])

            data_cog.append((
                "".join([c if c in vocab_ort else UNK_SYMBOL for c in word["word"]]),
                token_ipa,
                # set to special "multi" language
                "multi",
                f"cognate_{cognate_i}_{word_i}",
                " ".join(ipa2arp("".join(token_ipa_raw))),
            ))

    # no need to sort by languages here because we're adding a new one, "multi"
    with open("data/multi_3.tsv", "w") as f:
        for line in data:
            f.write("\t".join(line) + "\n")
        for line in data_cog:
            f.write("\t".join(line) + "\n")
