#!/usr/bin/env python3

import argparse
from main.utils import load_embd_data, load_multi_data
from eval_correlations import evaluate_correlations
from eval_retrieval import evaluate_retrieval
from eval_rhyme import evaluate_rhyme
import numpy as np

def evaluate_all(data_multi, data_embd, lang="all"):
    scores_all = []

    print("Correlations")
    output = evaluate_correlations(data_multi, data_embd)
    scores_all.append(max(output["pearson L2"][lang], output["pearson cos"][lang]))
    print("Retrieval")
    output = evaluate_retrieval(data_multi, data_embd)
    scores_all.append(max(output["rank L2"][lang], output["rank cos"][lang]))

    print("Rhyme")
    # currently only English is supported
    output = evaluate_rhyme(data_multi, data_embd)
    scores_all.append(output["dev"])

    return np.average(scores_all)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data-multi", default="data/multi.tsv")
    args.add_argument("-e", "--embd", default="computed/embd_bpemb.pkl")
    args.add_argument("-l", "--lang", default="all")
    args = args.parse_args()

    data_multi = load_multi_data(args.data_multi)
    data_embd = load_embd_data(args.embd)

    assert len(data_multi) == len(data_embd)

    output = evaluate_all(data_multi, data_embd, args.lang)
    print(f"Score (multi): {output:.4f}")

