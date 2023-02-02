#!/usr/bin/bash

mkdir -p computed/embd_rnn_metric_learning/

for FEATURES in "panphon" "tokenipa" "tokenort"; do
    SIGNATURE="join_${FEATURES}"
    sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG.pkl\" \
                --output \"computed/embd_rnn_metric_learning/${FEATURES}.pkl\" \
            ;"

    # the lang order here is super important
    SIGNATURE="join_${FEATURES}_f${LANGFROM}"
    # for LANGFROM in 'multi' 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
    for LANGFROM in 'multi'; do
        sbatch --time=00-04 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG_f${LANGFROM}.pkl\" \
                --output \"computed/embd_rnn_metric_learning/${FEATURES}_f${LANGFROM}.pkl\" \
            ;"
    done;
done;