#!/usr/bin/bash

mkdir -p computed/embd_rnn_metric_learning/size/

for SIZE in "1" "5" "10" "50" "100" "150" "200"; do
# the lang order here is super important
for FEATURES in "panphon"; do
    SIGNATURE="join_${FEATURES}_s${SIZE}"
    sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG_s${SIZE}.pkl\" \
                --output \"computed/embd_rnn_metric_learning/size/${FEATURES}_s${SIZE}.pkl\" \
            ;"
done;
done;