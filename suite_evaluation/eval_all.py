#!/usr/bin/env python3

import argparse
import json
from main.utils import load_embd_data, load_multi_data
from eval_correlations import evaluate_correlations
from eval_retrieval import evaluate_retrieval
from eval_rhyme import evaluate_rhyme
from eval_analogy import evaluate_analogy
from eval_human_similarity import evaluate_human_similarity
from eval_cognate import evaluate_cognate
import numpy as np


def evaluate_all(data_multi_all, data_embd, lang="all", jobs=20):
    scores_all = {}
    data_multi = [
        (x, y) for x, y in zip(data_multi_all, data_embd)
        if x["purpose"] == "main"
    ]
    data_multi_all_all = [
        (x, y) for x, y in zip(data_multi_all, data_embd)
    ]

    print("Human similarity")
    # currently only English is supported
    data_multi_hs = [
        (x, y) for x, y in zip(data_multi_all, data_embd)
        if x["purpose"] == "human_similarity"
    ]
    output = evaluate_human_similarity(data_multi_hs)
    scores_all["human_similarity"] = max(
        output["pearson L2"], output["pearson cos"], output["pearson IP"],
    )

    print("Correlations")
    output = evaluate_correlations(data_multi, jobs=jobs)
    scores_all["correlation"] = max(
        output["pearson L2"][lang], output["pearson cos"][lang]
    )

    print("Retrieval")
    output = evaluate_retrieval(data_multi, jobs=jobs)
    scores_all["retrieval"] = max(
        output["rank L2"][lang], output["rank cos"][lang]
    )

    print("Sound analogies")
    data_multi_analogy = [
        (x, y) for x, y in zip(data_multi_all, data_embd)
        if x["purpose"] == "analogy"
    ]
    output = evaluate_analogy(
        data_multi_analogy
    )
    scores_all["analogy"] = output["all"]

    print("Rhyme")
    # currently only English is supported
    output = evaluate_rhyme(data_multi)
    scores_all["rhyme"] = output["dev"]

    print("Cognate")
    output = evaluate_cognate(data_multi_all_all)
    scores_all["cognate"] = output["dev"]

    scores_all["overall"] = np.average(list(scores_all.values()))
    return scores_all["overall"], scores_all


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-e", "--embd", default="computed/embd_bpemb.pkl")
    args.add_argument("-l", "--lang", default="all")
    args = args.parse_args()

    data_multi = load_multi_data(purpose_key="all")
    data_embd = [np.array(x) for x in load_embd_data(args.embd)]

    if len(data_multi) != len(data_embd):
        print(f"Length mismatch ({len(data_multi)}, {len(data_embd)}), the suite will continue but the results will not be valid.")

    score, scores_all = evaluate_all(data_multi, data_embd, args.lang)
    for key, val in scores_all.items():
        print(f"{key}: {val:.4f}")
    print(f"JSON1!{json.dumps(scores_all)}")
    print(f"Score (overall): {score:.4f}")
