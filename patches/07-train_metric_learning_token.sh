#!/usr/bin/bash

mkdir -p computed/models

for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
    echo "$LANG tokenort"
    ./models/metric_learning/train.py \
        --data "data/multi.tsv" \
        --lang ${LANG} \
        --save-model-path "computed/models/rnn_metric_learning_tokenort_${LANG}.pt" \
        --number-thousands 200 \
        --target-metric "l2" \
        --epochs 20 \
        --features tokenort \
    ;
done;

for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
    echo "$LANG tokenipa"
    ./models/metric_learning/train.py \
        --data "data/multi.tsv" \
        --lang ${LANG} \
        --save-model-path "computed/models/rnn_metric_learning_tokenipa_${LANG}.pt" \
        --number-thousands 200 \
        --target-metric "l2" \
        --epochs 20 \
        --features tokenipa \
    ;
done;