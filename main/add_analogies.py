#!/usr/bin/env python3

import random
from collections import defaultdict
from functools import lru_cache
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
    def __init__(self, symbols, tokens, num_perturbations=1):
        self.single_perturbation_pairs = self.find_single_perturbation_pairs(symbols)
        self.all_tokens = tokens
        self.num_perturbations = num_perturbations
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

    def generate_analogy(self, w1, num_perturbations):
        # w1 is a randomly sampled real word
        # sample one phoneme ph1 from w1, and sample two perturbations of the same kind, one of which uses ph1
        # for example, if w1 contains t, the two perturbations might be: t <-> d and f <-> v
        # in the same position of t in w1, w2 gets d, w3 gets f, and w4 gets v
        w1_segs = ft.ipa_segs(w1)
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
            w2_char, perturb_type, plus_minus = self.random.choice(perturbations)
            w3_char, w4_char = w2_char, w2_char
            while w3_char == w2_char or w4_char == w2_char:
                w3_char, w4_char = self.random.choice(self.single_perturbation_pairs[perturb_type])
                # TODO: will we need to choose w3 and w4 so that they have the same syl feature as w1 and w2?
                # TODO: because if not, we get tuples like ulm	ylm	ŋlm	ɲlm
            if plus_minus == '-':
                w3_char, w4_char = w4_char, w3_char

            w2_segs[phoneme_idx] = self.random.choice(w2_char.split('/'))
            w3_segs[phoneme_idx] = self.random.choice(w3_char.split('/'))
            w4_segs[phoneme_idx] = self.random.choice(w4_char.split('/'))

        w2 = ''.join(w2_segs)
        w3 = ''.join(w3_segs)
        w4 = ''.join(w4_segs)
        return w2, w3, w4

    def run(self, num_analogies):
        output = []
        for i in range(num_analogies):
            res = None
            while res is None:
                w1 = ''
                while not (3 <= len(w1) <= 8):
                    w1 = self.random.choice(self.all_tokens)
                res = self.generate_analogy(w1, self.num_perturbations)

            w2, w3, w4 = res
            output.append([w1, w2, w3, w4])
        return output


    def find_single_perturbation_pairs(self, symbols):
        edges = [[] for _ in FEATURE_NAMES]

        feature_vectors = defaultdict(list)
        for x in symbols:
            feature_vectors[tuple(ft.fts(x).numeric())].append(x)
        for feat_i, feat_name in enumerate(FEATURE_NAMES):
            for vec, symb in feature_vectors.items():
                if vec[feat_i] == -1:
                    partner = list(vec)
                    partner[feat_i] = 1
                    partner = tuple(partner)
                    if partner in feature_vectors:
                        partner_symb = feature_vectors[partner]
                        symb_str, partner_symb_str = '/'.join(symb), '/'.join(partner_symb)
                        edges[feat_i].append((symb_str, partner_symb_str))
        return edges

def get_analogies(data):
    with open(f"data/vocab/ipa_multi.txt") as f:
        vocab_ipa_multi = f.read().split()

    output = []

    data_local = [
        x[1] for x in data
    ]

    for num_perturbations in [1, 2]:
        analogy_model = PhonemeAnalogy(
            vocab_ipa_multi, data_local, num_perturbations=num_perturbations
        )
        analogies = analogy_model.run(100)
        output += analogies

    return output

if __name__ == '__main__':
    data = load_multi_data(path="data/multi_0.tsv")
    data_analogies = []

    for lang in LANGS:
        data_local = [x for x in data if x[2] == lang]
        # this will run it across all languages
        output = get_analogies(data_local)
        for analogy in output:
            # append only IPA
            # TODO: figure out how to permute also the character?
            data_analogies.append(("", analogy[0], lang, "analogy", ""))
            data_analogies.append(("", analogy[1], lang, "analogy", ""))
            data_analogies.append(("", analogy[2], lang, "analogy", ""))
            data_analogies.append(("", analogy[3], lang, "analogy", ""))
    

    print(len(data), len(data_analogies))
    with open("data/multi.tsv", "w") as f:
        for line in data:
            f.write("\t".join([line[0], line[1], line[2], "main", line[3]])+"\n")
        for line in data_analogies:
            f.write("\t".join(line)+"\n")