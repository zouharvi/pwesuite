#!/usr/bin/bash

mkdir -p computed/models

for FEATURES in "panphon" "tokenort" "tokenipa"; do
    for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
        SIGNATURE="train_rnn_${FEATURES}_${LANG}"
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
            --job-name="${SIGNATURE}" \
            --output="logs/${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/metric_learning/train.py \
                    --data \"data/multi.tsv\" \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
                    --number-thousands 200 \
                    --target-metric \"l2\" \
                    --features ${FEATURES} \
                    --epochs 20 \
                ;"
    done;
    
    for LANG in 'multi'; do
        SIGNATURE="train_rnn_${FEATURES}_${LANG}"
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
            --job-name="${SIGNATURE}" \
            --output="logs/${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/metric_learning/train.py \
                    --data \"data/multi.tsv\" \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
                    --number-thousands 400 \
                    --target-metric \"l2\" \
                    --features ${FEATURES} \
                    --epochs 20 \
                ;"
    done;
done;

# FEATURES=tokenort
# LANG=multi
# python3 \
#     ./models/metric_learning/train.py \
#         --data "data/multi.tsv" \
#         --lang ${LANG} \
#         --save-model-path "computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt" \
#         --number-thousands 200 \
#         --target-metric "l2" \
#         --features ${FEATURES} \
#         --epochs 20 \
#     ;