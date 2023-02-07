#!/usr/bin/env python3

import argparse
from main.utils import load_embd_data, load_multi_data, LANGS
from eval_all import evaluate_all
import numpy as np
import json
import multiprocessing as mp
from itertools import product

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data-multi", default="data/multi.tsv")
    args.add_argument("-e", "--embd", default="computed/embd_rnn_metric_learning/panphon_fLANG.pkl")
    args = args.parse_args()

    data_multi = load_multi_data(args.data_multi, purpose_key="all")
    output_scores = {}
    output_all = {}

    lang_pairs = list(product(LANGS, repeat=2))

    def evaluate_single_pair(langs):
        langFrom, lang = langs

        fname = args.embd.replace("LANG", langFrom)
        data_embd = load_embd_data(fname)
        assert len(data_multi) == len(data_embd)

        print(langFrom, "->", lang)
        score, scores_all = evaluate_all(data_multi, data_embd, lang, jobs=5)
        return score, scores_all

    # do all tasks in parallel
    with mp.Pool(10) as pool:
        output_raw = pool.map(
            evaluate_single_pair, lang_pairs
        )
    for (langFrom, lang), (score, scores_all) in zip(lang_pairs, output_raw):
        output_scores[f"{langFrom}-{lang}"] = score
        output_all[f"{langFrom}-{lang}"] = scores_all        

    print(f"JSON1!{json.dumps(output_scores)}")
    print(f"JSON2!{json.dumps(output_all)}")
    
    for task_key in output_all["en-en"].keys():
        scores = [v[task_key] for k, v in output_all.items() if k.split("-")[0] != k.split("-")[1]]
        print(f"(mismatch) {task_key}: {np.average(scores):.4f}")

    output_mismatch = {k:v for k, v in output_scores.items() if k.split("-")[0] != k.split("-")[1]}
    print(f"Score (mismatch): {np.average(list(output_mismatch.values())):.4f}")
    
    for task_key in output_all["en-en"].keys():
        scores = [v[task_key] for k, v in output_all.items() if k.split("-")[0] == k.split("-")[1]]
        print(f"(match) {task_key}: {np.average(scores):.4f}")

    output_match = {k:v for k, v in output_scores.items() if k.split("-")[0] == k.split("-")[1]}
    print(f"Score (match): {np.average(list(output_match.values())):.4f}")