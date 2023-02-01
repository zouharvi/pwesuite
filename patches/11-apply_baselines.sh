#!/usr/bin/bash

for MODEL in "bert"; do
    SIGNATURE="apply_baseline_${MODEL}"
    sbatch --time=01-00 --ntasks=30 --mem-per-cpu=1G --gpus=1 \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./models/baselines/${MODEL}_embd.py \
        ;"

    # SIGNATURE="eval_all_rnn_${FEATURES}_multi"
    # sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
    #     --job-name="${SIGNATURE}" \
    #     --output="logs/${SIGNATURE}.log" \
    #     --wrap="\
    #         ./suite_evaluation/eval_all.py \
    #             --embd \"computed/embd_rnn_metric_learning/${FEATURES}_fmulti.pkl\" \
    #         ;"

    # SIGNATURE="eval_mismatch_rnn_${FEATURES}"
    # sbatch --time=01-00 --ntasks=100 --mem-per-cpu=500M  \
    #     --job-name="${SIGNATURE}" \
    #     --output="logs/${SIGNATURE}.log" \
    #     --wrap="\
    #         ./suite_evaluation/eval_mismatch.py \
    #             --embd \"computed/embd_rnn_metric_learning/${FEATURES}_fLANG.pkl\" \
    #         ;"
done;