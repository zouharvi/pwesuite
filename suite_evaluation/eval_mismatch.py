#!/usr/bin/env python3

import argparse
from main.utils import load_embd_data, load_multi_data
from eval_all import evaluate_all
import numpy as np
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data-multi", default="data/multi.tsv")
    args.add_argument("-e", "--embd", default="computed/embd_rnn_metric_learning_fLANG.pkl")
    args = args.parse_args()

    data_multi = load_multi_data(args.data_multi)
    output = {}

    for langFrom in ['en', 'am', 'bn', 'uz', 'pl', 'es', 'sw']:
        fname = args.embd.replace("LANG", langFrom)
        data_embd = load_embd_data(fname)
        assert len(data_multi) == len(data_embd)

        for lang in ['en', 'am', 'bn', 'uz', 'pl', 'es', 'sw']:
            print(langFrom, "->", lang)
            output[f"{langFrom}-{lang}"] = evaluate_all(data_multi, data_embd, lang)

    print(f"JSON!{json.dumps(output)}")
    output_mismatch = {k:v for k, v in output.items() if k.split("-")[0] != k.split("-")[1]}
    print(f"Score (mismatch): {np.average(list(output_mismatch.values())):.4f}")
    
    output_match = {k:v for k, v in output.items() if k.split("-")[0] == k.split("-")[1]}
    print(f"Score (match): {np.average(list(output_match.values())):.4f}")