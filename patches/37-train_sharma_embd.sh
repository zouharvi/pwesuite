#!/usr/bin/bash

mkdir -p computed/tmp/

# for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de' 'multi'; do
for LANG in 'multi'; do
    SIGNATURE="sharma_embd_${LANG}"
    sbatch --time=07-00 --ntasks=60 --mem-per-cpu=4G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            python3 ./models/sharma/train_embd_all.py \
                --lang ${LANG} \
                --ntasks 30 \
                --batch-size 6000 \
            ;"
done;