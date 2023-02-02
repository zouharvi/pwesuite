#!/usr/bin/bash

mkdir -p computed/embd_rnn_metric_learning/

# the lang order here is super important
for FEATURES in "panphon" "tokenort" "tokenipa"; do
    SIGNATURE="join_rnn_autoencoder_${FEATURES}"
    sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_autoencoder_${FEATURES}_LANG.pkl\" \
                --output \"computed/embd_rnn_autoencoder/${FEATURES}.pkl\" \
            ;"

done;