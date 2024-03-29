#!/usr/bin/env python3

import numpy as np
import argparse
from main.utils import load_multi_data, load_embd_data
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from create_dataset.add_cognates import get_cognates

def evaluate_cognate(data_multi_all):
    data_multi = [
        # embd, token_ort, token_ipa, token_pron
        (x[5], x[0], x[1], x[4]) for x in data_multi_all
        # we have pronunciation information only for English
        if x[2] == "multi"
    ]

    word2embd = {x[1]:x[0] for x in data_multi}
    cognates = get_cognates()

    data_task = []

    for w1, w_pos, w_neg in cognates:
        # make sure everything is numpy
        embd1 = np.array(word2embd[w1["word"]])
        embd2 = np.array(word2embd[w_pos["word"]])
        embd3 = np.array(word2embd[w_neg["word"]])

        data_task.append((np.concatenate((embd1, embd2)), True))
        data_task.append((np.concatenate((embd1, embd3)), False))

    data_dev, data_train = train_test_split(data_task, test_size=500, random_state=0)

    model = MLPClassifier(
        hidden_layer_sizes=(10, 5, 5),
        random_state=0,
        learning_rate="adaptive",
    )
    model.fit(
        [x[0] for x in data_train],
        [x[1] for x in data_train],
    )
    acc_train = model.score(
        [x[0] for x in data_train],
        [x[1] for x in data_train],
    )
    acc_dev = model.score(
        [x[0] for x in data_dev],
        [x[1] for x in data_dev],
    )

    return {"train": max(acc_train, 1-acc_train), "dev": max(acc_dev, 1-acc_dev)}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data", default="data/multi.tsv")
    args.add_argument("-e", "--embd", default="computed/embd_bert.pkl")
    args = args.parse_args()

    data_embd = load_embd_data(args.embd)
    data_multi_all = load_multi_data(args.data, purpose_key="all")

    data_multi = [
        (*x, np.array(y)) for x, y in zip(data_multi_all, data_embd)
    ]

    output = evaluate_cognate(data_multi)

    print("Overall:")
    for key in output:
        print(f"{key}: {output[key]:.1%}")