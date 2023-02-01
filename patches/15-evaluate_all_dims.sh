#!/usr/bin/bash

for DIMS in "50" "100" "150" "200" "300" "500" "700"; do
for FEATURES in "panphon"; do
    SIGNATURE="eval_all_rnn_${FEATURES}_d${DIMS}"
    sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./suite_evaluation/eval_all.py \
                --embd \"computed/embd_rnn_metric_learning/dims/${FEATURES}_d${DIMS}.pkl\" \
            ;"
done;
done;