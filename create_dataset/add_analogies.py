#!/usr/bin/env python3

import collections
import os
import pickle
import random
from collections import defaultdict
from functools import lru_cache

import tqdm
from main.utils import load_multi_data, LANGS
import panphon

ft = panphon.FeatureTable()
FEATURE_NAMES = ft.fts('a').names

'''
Example command: 
python ./data/analogies/generate_analogies.py --lang_codes uz \
--vocab_file data/vocab_uz.txt --output_file data/analogies/uz100_2.txt \
--num_analogies 100 --num_perturbations 2
'''

# parser.add_argument('--num_analogies', type=int, help='size of the resulting analogy dataset')
# parser.add_argument('--num_perturbations', type=int,
#     help='number phonemes to perturb. '
#     'eg. sik : zik <=> ʂik : ʐik is one perturbation,'
#     '    sik : zix <=> ʂic : ʐiç is two perturbations.'
# )


class PhonemeAnalogy:
    def __init__(self, symbols, tokens):
        self.single_perturbation_pairs = self.find_single_perturbation_pairs(
            symbols)
        self.all_tokens = tokens
        self.ipa_to_char = collections.defaultdict(list)

        # TODO: paralelize
        for token_i in range(len(self.all_tokens)):
            token = self.all_tokens[token_i]
            char_orths = token[0]
            char_ipas = ft.ipa_segs(token[1])
            # store orth and ipa segments
            self.all_tokens[token_i] = (token[0], char_ipas)
            if len(char_orths) != len(char_ipas):
                continue
            for char_orth, char_ipa in zip(char_orths, char_ipas):
                self.ipa_to_char[char_ipa] = char_orth

        self.random = random.Random(0)

    @lru_cache(None)
    def has_perturbations(self, phoneme):
        return len(self.get_all_perturbations(phoneme)) > 0

    def get_perturbation(self, phoneme, feat_i):
        perturbations = set()
        for ph1, ph2 in self.single_perturbation_pairs[feat_i]:
            if phoneme in ph1.split("/"):
                perturbations.add((ph2, '+'))
            elif phoneme in ph2.split("/"):
                perturbations.add((ph1, '-'))
        return perturbations

    @lru_cache(None)
    def get_all_perturbations(self, phoneme):
        perturbations = []
        for feat_i in range(len(FEATURE_NAMES)):
            perturbs = self.get_perturbation(phoneme, feat_i)
            if len(self.single_perturbation_pairs[feat_i]) >= 2:
                # there is at least one other pair of the same type
                for p, pn in perturbs:
                    perturbations.append((p, feat_i, pn))
        return perturbations

    def ipa_segs_to_orth(self, ipa_segs):
        if not all(s in self.ipa_to_char for s in ipa_segs):
            return None
        else:
            return "".join([self.ipa_to_char[s] for s in ipa_segs])

    def generate_analogy(self, w1, num_perturbations):
        # w1 is a randomly sampled real word
        # sample one phoneme ph1 from w1, and sample two perturbations of the same kind, one of which uses ph1
        # for example, if w1 contains t, the two perturbations might be: t <-> d and f <-> v
        # in the same position of t in w1, w2 gets d, w3 gets f, and w4 gets v
        w1_orth, w1_segs = w1

        # abort if the characters aren't aligned
        if len(w1_orth) != len(w1_segs):
            return None

        w2_segs, w3_segs, w4_segs = w1_segs.copy(), w1_segs.copy(), w1_segs.copy()

        phoneme_idx_perturbed = set()
        for pi in range(num_perturbations):
            retry_counter = 0
            phoneme_idx = self.random.choice(range(len(w1_segs)))
            while (phoneme_idx in phoneme_idx_perturbed or not self.has_perturbations(w1_segs[phoneme_idx])):
                retry_counter += 1
                if retry_counter > 100:
                    return None
                phoneme_idx = self.random.choice(range(len(w1_segs)))
            phoneme_idx_perturbed.add(phoneme_idx)
            perturbations = self.get_all_perturbations(w1_segs[phoneme_idx])
            # e.g. w1[phoneme_idx] is z
            # perturbations is {('s', 8), ('d', 3), ('ð', 13)}
            w2_char, perturb_type, plus_minus = self.random.choice(
                perturbations)
            w3_char, w4_char = w2_char, w2_char
            while w3_char == w2_char or w4_char == w2_char:
                w3_char, w4_char = self.random.choice(
                    self.single_perturbation_pairs[perturb_type])
                # TODO: will we need to choose w3 and w4 so that they have the same syl feature as w1 and w2?
                # TODO: because if not, we get tuples like ulm	ylm	ŋlm	ɲlm
            if plus_minus == '-':
                w3_char, w4_char = w4_char, w3_char

            w2_segs[phoneme_idx] = self.random.choice(w2_char.split('/'))
            w3_segs[phoneme_idx] = self.random.choice(w3_char.split('/'))
            w4_segs[phoneme_idx] = self.random.choice(w4_char.split('/'))

        w1 = (self.ipa_segs_to_orth(w1_segs), ''.join(w1_segs))
        w2 = (self.ipa_segs_to_orth(w2_segs), ''.join(w2_segs))
        w3 = (self.ipa_segs_to_orth(w3_segs), ''.join(w3_segs))
        w4 = (self.ipa_segs_to_orth(w4_segs), ''.join(w4_segs))

        # remove if unable to convert
        if any([w[0] is None for w in [w1, w2, w3, w4]]):
            return None
        # remove if nonunique
        if len(set([w[0] for w in [w1, w2, w3, w4]])) != 4:
            return None

        return w1, w2, w3, w4

    def run(self, num_analogies, num_perturbations):
        output = []
        for i in tqdm.tqdm(range(num_analogies)):
            res = None
            while res is None:
                w1 = ("", "")
                while not (3 <= len(w1[0]) <= 8):
                    w1 = self.random.choice(self.all_tokens)
                res = self.generate_analogy(w1, num_perturbations)

            w1, w2, w3, w4 = res
            output.append([w1, w2, w3, w4])
        return output

    def find_single_perturbation_pairs(self, symbols):
        edges = [[] for _ in FEATURE_NAMES]

        feature_vectors = defaultdict(list)
        for x in symbols:
            if ft.fts(x) is not None:
                feature_vectors[tuple(ft.fts(x).numeric())].append(x)
        for feat_i, feat_name in enumerate(FEATURE_NAMES):
            for vec, symb in feature_vectors.items():
                if vec[feat_i] == -1:
                    partner = list(vec)
                    partner[feat_i] = 1
                    partner = tuple(partner)
                    if partner in feature_vectors:
                        partner_symb = feature_vectors[partner]
                        symb_str, partner_symb_str = '/'.join(
                            symb), '/'.join(partner_symb)
                        edges[feat_i].append((symb_str, partner_symb_str))
        return edges


def get_analogies(data, lang):
    CACHE_PATH = f"data/cache/analogies_{lang}.pkl"
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    with open(f"data/vocab/ipa_multi.txt") as f:
        vocab_ipa_multi = f.read().split()

    output = []

    analogy_model = PhonemeAnalogy(
        vocab_ipa_multi, data,
    )
    for num_perturbations in [1, 2]:
        analogies = analogy_model.run(100, num_perturbations=num_perturbations)
        output += analogies

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(output, f)

    return output


if __name__ == '__main__':
    data = load_multi_data(path="data/multi_0.tsv")
    data_analogies = []

    for lang in LANGS:
        data_local = [x for x in data if x[2] == lang]
        print(lang, "prev", len(data_local))
        # this will run it across all languages
        output = get_analogies(data_local, lang)
        for analogy in output:
            # append only IPA
            for token_ort, token_ipa in analogy:
                data_analogies.append((
                    token_ort, token_ipa, lang, "analogy", ""
                ))

    print("Adding", len(data_analogies), "words")
    data_langs = collections.defaultdict(list)
    for line in data+data_analogies:
        data_langs[line[2]].append(line)

    with open("data/multi_1.tsv", "w") as f:
        for lang in LANGS:
            print(lang, "now", len(data_langs[lang]))
            for line in data_langs[lang]:
                f.write("\t".join(line) + "\n")
