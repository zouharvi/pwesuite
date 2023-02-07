#!/usr/bin/bash

mkdir -p computed/tmp

for FEATURES in "panphon" "tokenipa" "tokenort"; do
for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
    SIGNATURE="apply_rnn_autoencoder_${FEATURES}_${LANG}"
    sbatch --time=00-01 --ntasks=8 --mem-per-cpu=4G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./models/autoencoder/apply.py \
                --data \"data/multi.tsv\" \
                --lang ${LANG} \
                --model-path \"computed/models/rnn_autoencoder_${FEATURES}_${LANG}.pt\" \
                --features ${FEATURES} \
                --output \"computed/tmp/multi_autoencoder_${FEATURES}_${LANG}.pkl\" \
            ;"

done;
done;