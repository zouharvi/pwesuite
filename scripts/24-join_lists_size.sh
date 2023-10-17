#!/usr/bin/bash

mkdir -p computed/embd_rnn_metric_learning/size/

for SEED in "0" "1" "2" "3" "4"; do
for SIZE in "1" "5" "10" "50" "100" "150" "200"; do
for FEATURES in "panphon"; do
    SIGNATURE="join_${FEATURES}_s${SIZE}_s${SEED}"
    sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./scripts/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG_s${SIZE}_s${SEED}.pkl\" \
                --output \"computed/embd_rnn_metric_learning/size/${FEATURES}_s${SIZE}_s${SEED}.pkl\" \
            ;"
done;
done;
done;