#!/usr/bin/bash

for SIZE in "1" "5" "10" "50" "100" "150" "200"; do
for FEATURES in "panphon"; do
    SIGNATURE="eval_all_rnn_${FEATURES}_s${SIZE}"
    sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./suite_evaluation/eval_all.py \
                --embd \"computed/embd_rnn_metric_learning/size/${FEATURES}_s${SIZE}.pkl\" \
            ;"
done;
done;