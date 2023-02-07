#!/usr/bin/bash

mkdir -p computed/tmp

for FEATURES in "panphon" "tokenipa" "tokenort"; do
for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de' 'multi'; do
# for LANG in 'sw'; do
    SIGNATURE="apply_rnn_${FEATURES}_${LANG}"
    sbatch --time=00-01 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./models/metric_learning/apply.py \
                --data \"data/multi.tsv\" \
                --lang ${LANG} \
                --model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
                --features ${FEATURES} \
                --output \"computed/tmp/multi_${FEATURES}_${LANG}.pkl\" \
            ;"

        # for LANGFROM in 'uz' ; do
        for LANGFROM in 'multi' 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
            SIGNATURE="apply_rnn_${FEATURES}_${LANG}_f${LANGFROM}"
            sbatch --time=01-00 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
                --job-name="${SIGNATURE}" \
                --output="logs/${SIGNATURE}.log" \
                --wrap="\
                    ./models/metric_learning/apply.py \
                        --data \"data/multi.tsv\" \
                        --lang ${LANG} \
                        --model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANGFROM}.pt\" \
                        --features ${FEATURES} \
                        --output \"computed/tmp/multi_${FEATURES}_${LANG}_f${LANGFROM}.pkl\" \
                    ;"
        done
done;
done;