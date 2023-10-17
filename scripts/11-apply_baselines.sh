#!/usr/bin/bash

for MODEL in "bert"; do
    SIGNATURE="apply_baseline_${MODEL}"
    sbatch --time=01-00 --ntasks=30 --mem-per-cpu=1G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./models/baselines/${MODEL}_embd.py \
        ;"
done;

for MODEL in "bert"; do
    SIGNATURE="eval_baseline_${MODEL}"
    sbatch --time=01-00 --ntasks=30 --mem-per-cpu=1G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./suite_evaluation/eval_all.py -e computed/embd_baseline/bert.pkl \
        ;"
done;


# for MODEL in "bpemb" "fasttext" "instructor"; do
#     python3 ./models/baselines/${MODEL}_embd.py
# done;