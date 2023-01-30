#!/usr/bin/bash

mkdir -p computed/tmp

for FEATURES in "panphon" "tokenort" "tokenipa"; do
for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
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

        for LANGFROM in 'multi' 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
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

mkdir -p computed/embd_rnn_metric_learning/

# the lang order here is super important
for FEATURES in "panphon" "tokenort" "tokenipa"; do
    SIGNATURE="join_${FEATURES}_${LANG}"
    sbatch --time=01-00 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG.pkl\" \
                --output \"computed/embd_rnn_metric_learning/${FEATURES}.pkl\" \
            ;"

    SIGNATURE="join_${FEATURES}_${LANG}_f${LANGFROM}"
    for LANGFROM in 'multi' 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG_f${LANGFROM}.pkl\" \
                --output \"computed/embd_rnn_metric_learning/${FEATURES}_f${LANGFROM}.pkl\" \
            ;"
    done;
done;