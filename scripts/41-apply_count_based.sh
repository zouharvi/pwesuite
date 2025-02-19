#!/usr/bin/bash

for DIM in 50 100 150 200 250 300 350 400 450 500 550 600; do
    echo "Running ${DIM}";
    ./models/count_based/apply.py \
        --features token_ort \
        --vectorizer tfidf \
        --nopca \
        --force-dim $DIM \
    > /dev/null;
done


for DIM in 50 100 150 200 250 300 350 400 450 500 550 600; do
    echo "Running ${DIM}";
    ./suite_evaluation/eval_all.py \
        --embd "computed/embd_other/count_token_ort_tfidf_nopca_dim${DIM}.pkl" \
    2>/dev/null | grep "(overall)";
done