#!/usr/bin/bash

mkdir -p computed/models/dims

for SEED in "0" "1" "2" "3" "4"; do
for DIMS in "50" "100" "150" "200" "300" "500" "700"; do
    mkdir -p "computed/models/dims/${DIMS}"
for FEATURES in "panphon"; do
    for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de' 'multi'; do
        SIGNATURE="train_rnn_${FEATURES}_${LANG}_d${DIMS}_s${SEED}"
        sbatch --time=01-00 --ntasks=12 --mem-per-cpu=4G --gpus=1 \
            --job-name="${SIGNATURE}" \
            --output="logs/${SIGNATURE}.log" \
            --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
                ./models/metric_learning/train.py \
                    --lang ${LANG} \
                    --save-model-path \"computed/models/dims/${DIMS}/rnn_metric_learning_${FEATURES}_${LANG}_s${SEED}.pt\" \
                    --number-thousands 200 \
                    --target-metric \"l2\" \
                    --features ${FEATURES} \
                    --dimension ${DIMS} \
                    --epochs 20 \
                    --seed ${SEED} \
                ;"
    done;
done;
done;
done;

# FEATURES=token_ort
# LANG=multi
# python3 \
#     ./models/metric_learning/train.py \
#         --lang ${LANG} \
#         --save-model-path "computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt" \
#         --number-thousands 200 \
#         --target-metric "l2" \
#         --features ${FEATURES} \
#         --epochs 20 \
#     ;

# FEATURES=token_ort
# for LANG in 'multi'; do
#     SIGNATURE="train_rnn_${FEATURES}_${LANG}"
#     sbatch --time=01-00 --ntasks=15 --mem-per-cpu=5G --gpus=1 \
#         --job-name="${SIGNATURE}" \
#         --output="logs/${SIGNATURE}.log" \
#         --wrap="CUDA_VISIBLE_DEVICES=0 python3 \
#             ./models/metric_learning/train.py \
#                 --lang ${LANG} \
#                 --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
#                 --number-thousands 12000 \
#                 --target-metric \"l2\" \
#                 --features ${FEATURES} \
#                 --epochs 20 \
#             ;"
# done;