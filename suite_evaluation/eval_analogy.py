#!/usr/bin/env python3

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import argparse
from main.utils import load_embd_data, load_multi_data, LANGS
from create_dataset.add_analogies import get_analogies


def evaluate_analogy_single_lang(data_local, data_local_analogies, lang):
    analogies = get_analogies(data_local, lang)

    ipa_to_i = {x[1]: i for i, x in enumerate(data_local_analogies)}
    embd_all = [x[5] for x in data_local_analogies]

    analogies_embd_indicies = []
    for analogy in analogies:
        if all([w[1] in ipa_to_i for w in analogy]):
            # 0 is ort, 1 is ipa
            embd_a = data_local_analogies[ipa_to_i[analogy[0][1]]][5]
            embd_b = data_local_analogies[ipa_to_i[analogy[1][1]]][5]
            embd_c = data_local_analogies[ipa_to_i[analogy[2][1]]][5]
            embd_d = embd_b - embd_a + embd_c
            analogies_embd_indicies.append((ipa_to_i[analogy[3][1]], embd_d))

    dists_all = euclidean_distances(
        [x[1] for x in analogies_embd_indicies],
        embd_all
    )

    ranks = []
    for dists, (index_d, _embd) in zip(dists_all, analogies_embd_indicies):
        dists = sorted(
            range(len(dists)),
            key=lambda i: dists[i], reverse=False
        )
        rank = dists.index(index_d)
        # hit if rank==0 or rank==1
        ranks.append(rank <= 1)

    return np.average(ranks)


def evaluate_analogy(data_multi, data_multi_analogies, jobs=20):
    output = {}
    for lang in LANGS:
        data_local = [
            x for x in data_multi
            if x[2] == lang
        ]
        if len(data_local) == 0:
            continue
        data_local_analogies = [
            x for x in data_multi_analogies
            if x[2] == lang
        ]
        output[lang] = evaluate_analogy_single_lang(
            data_local, data_local_analogies, lang
        )

    output["all"] = np.average(list(output.values()))
    return output


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data-multi", default="data/multi.tsv")
    args.add_argument(
        "-e", "--embd", default="computed/embd_rnn_metric_learning/panphon.pkl")
    args = args.parse_args()

    data_multi = load_multi_data(args.data_multi)
    data_multi_all = load_multi_data(
        args.data_multi, purpose_key="all"
    )
    data_embd = load_embd_data(args.embd)

    assert len(data_multi_all) == len(data_embd)

    data_multi_all = [
        (*x, y) for x, y in zip(data_multi_all, data_embd)
        if x[3] == "analogy"
    ]

    # TODO: why is it so high for Swahili?
    output = evaluate_analogy(data_multi, data_multi_all)

    print("Overall:", f"{output['all']:.3f}")
