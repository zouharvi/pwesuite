#!/usr/bin/bash

mkdir -p computed/embd_other/

SIGNATURE="join_sharma"
sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
    --job-name="${SIGNATURE}" \
    --output="logs/${SIGNATURE}.log" \
    --wrap="\
        ./patches/06-join_lists.py \
            --input \"computed/tmp/sharma_embd_LANG.pkl\" \
            --output \"computed/embd_other/sharma.pkl\" \
        ;"


SIGNATURE="eval_baseline_sharma"
sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
    --job-name="${SIGNATURE}" \
    --output="logs/${SIGNATURE}.log" \
    --wrap="\
        ./suite_evaluation/eval_all.py \
            --embd \"computed/embd_other/sharma.pkl\" \
        ;"