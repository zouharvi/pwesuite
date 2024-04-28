#!/usr/bin/bash

mkdir -p computed/models/size

for SEED in "0" "1" "2" "3" "4"; do
for SIZE in "1" "5" "10" "50" "100" "150" "200"; do
mkdir -p "computed/models/size/${SIZE}"
for FEATURES in "panphon"; do
    for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de' 'multi'; do
        SIGNATURE="train_rnn_${FEATURES}_${LANG}_s${SIZE}_s${SEED}"
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
            --job-name="${SIGNATURE}" \
            --output="logs/${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/metric_learning/train.py \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/size/${SIZE}/rnn_metric_learning_${FEATURES}_${LANG}_s${SEED}.pt\" \
                    --number-thousands ${SIZE} \
                    --target-metric \"l2\" \
                    --features ${FEATURES} \
                    --epochs 20 \
                    --seed ${SEED} \
                ;"
    done;
done;
done;
done;