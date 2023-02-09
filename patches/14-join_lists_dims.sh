#!/usr/bin/bash

mkdir -p computed/embd_rnn_metric_learning/dims/

for SEED in "0" "1" "2" "3" "4"; do
for DIMS in "50" "100" "150" "200" "300" "500" "700"; do
for FEATURES in "panphon"; do
    SIGNATURE="join_${FEATURES}_d${DIMS}_s${SEED}"
    sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG_d${DIMS}_s${SEED}.pkl\" \
                --output \"computed/embd_rnn_metric_learning/dims/${FEATURES}_d${DIMS}_s${SEED}.pkl\" \
            ;"
done;
done;
done;