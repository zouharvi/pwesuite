#!/usr/bin/bash

mkdir -p computed/embd_rnn_metric_learning/dims/

for DIMS in "50" "100" "150" "200" "300" "500" "700"; do
# the lang order here is super important
for FEATURES in "panphon"; do
    SIGNATURE="join_${FEATURES}_d${DIMS}"
    sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
        --job-name="${SIGNATURE}" \
        --output="logs/${SIGNATURE}.log" \
        --wrap="\
            ./patches/06-join_lists.py \
                --input \"computed/tmp/multi_${FEATURES}_LANG_d${DIMS}.pkl\" \
                --output \"computed/embd_rnn_metric_learning/dims/${FEATURES}_d${DIMS}.pkl\" \
            ;"
done;
done;

# the lang order here is super important
# for FEATURES in "panphon" "tokenipa" "tokenort"; do
#     SIGNATURE="join_${FEATURES}"
#     sbatch --time=00-04 --ntasks=8 --mem-per-cpu=4G \
#         --job-name="${SIGNATURE}" \
#         --output="logs/${SIGNATURE}.log" \
#         --wrap="\
#             ./patches/06-join_lists.py \
#                 --input \"computed/tmp/multi_${FEATURES}_LANG.pkl\" \
#                 --output \"computed/embd_rnn_metric_learning/${FEATURES}.pkl\" \
#             ;"
# done;