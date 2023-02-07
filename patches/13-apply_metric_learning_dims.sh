#!/usr/bin/bash

mkdir -p computed/tmp

for DIMS in "50" "100" "150" "200" "300" "500" "700"; do
for FEATURES in "panphon"; do
for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
    SIGNATURE="apply_rnn_${FEATURES}_${LANG}_d${DIMS}"
    sbatch --time=00-01 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./models/metric_learning/apply.py \
                --data \"data/multi.tsv\" \
                --lang ${LANG} \
                --model-path \"computed/models/dims/${DIMS}/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
                --features ${FEATURES} \
                --dimension ${DIMS} \
                --output \"computed/tmp/multi_${FEATURES}_${LANG}_d${DIMS}.pkl\" \
            ;"
done;
done;
done;