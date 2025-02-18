#!/usr/bin/bash

mkdir -p computed/models

for FEATURES in "panphon" "token_ipa" "token_ort"; do
    for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
        SIGNATURE="train_rnn_${FEATURES}_${LANG}"
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
            --job-name="${SIGNATURE}" \
            --output="logs/${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/metric_learning/train.py \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
                    --number-thousands 200 \
                    --target-metric \"l2\" \
                    --features ${FEATURES} \
                    --epochs 20 \
                ;"
    done;
    
    # for LANG in 'all' 'multi'; do
    for LANG in 'all'; do
        sbatch --time=01-00 --ntasks=15 --mem-per-cpu=5G --gpus=1 \
            --job-name="train_rnn_${FEATURES}_${LANG}" \
            --output="logs/%x.out" --error="logs/%x.err" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/metric_learning/train.py \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
                    --number-thousands 10000 \
                    --target-metric \"l2\" \
                    --features ${FEATURES} \
                    --epochs 20 \
                ;"
    done;
done;