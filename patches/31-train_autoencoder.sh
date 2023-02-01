#!/usr/bin/bash

mkdir -p computed/models

for FEATURES in "tokenipa"; do
    for LANG in 'sw'; do
        SIGNATURE="train_rnn_autoencoder_${FEATURES}_${LANG}"
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
            --job-name="${SIGNATURE}" \
            --output="logs/${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/autoencoder/train.py \
                    --data \"data/multi.tsv\" \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/rnn_autoencoder_${FEATURES}_${LANG}.pt\" \
                    --number-thousands 200 \
                    --features ${FEATURES} \
                    --epochs 20 \
                ;"
    done;
    
done;