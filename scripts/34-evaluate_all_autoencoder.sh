#!/usr/bin/bash

for FEATURES in "panphon" "token_ipa" "token_ort"; do
    SIGNATURE="eval_all_rnn_autoencoder_${FEATURES}"
    sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./suite_evaluation/eval_all.py \
                --embd \"computed/embd_rnn_autoencoder/${FEATURES}.pkl\" \
            ;"
done;