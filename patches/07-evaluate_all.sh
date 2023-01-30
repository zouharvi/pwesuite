#!/usr/bin/bash

for FEATURES in "panphon" "tokenort" "tokenipa"; do
    # SIGNATURE="eval_all_rnn_${FEATURES}"
    # sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
    #     --job-name="${SIGNATURE}" \
    #     --output="logs/${SIGNATURE}.log" \
    #     --wrap="\
    #         ./suite_evaluation/eval_all.py \
    #             --embd \"computed/embd_rnn_metric_learning/${FEATURES}.pkl\" \
    #         ;"

    # SIGNATURE="eval_all_rnn_${FEATURES}_multi"
    # sbatch --time=00-01 --ntasks=30 --mem-per-cpu=1G  \
    #     --job-name="${SIGNATURE}" \
    #     --output="logs/${SIGNATURE}.log" \
    #     --wrap="\
    #         ./suite_evaluation/eval_all.py \
    #             --embd \"computed/embd_rnn_metric_learning/${FEATURES}_fmulti.pkl\" \
    #         ;"

    SIGNATURE="eval_mismatch_rnn_${FEATURES}"
    sbatch --time=01-00 --ntasks=100 --mem-per-cpu=500M  \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./suite_evaluation/eval_mismatch.py \
                --embd \"computed/embd_rnn_metric_learning/${FEATURES}_fLANG.pkl\" \
            ;"
done;