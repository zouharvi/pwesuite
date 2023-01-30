#!/usr/bin/bash

mkdir -p computed/tmp

for LANG in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
    ./models/metric_learning/apply.py \
        --data "data/multi.tsv" \
        --lang ${LANG} \
        --model-path "computed/models/rnn_metric_learning_${LANG}.pt" \
        --output "computed/tmp/multi_${LANG}.pkl" \
    ;

    for LANGFROM in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
        ./models/metric_learning/apply.py \
            --data "data/multi.tsv" \
            --lang ${LANG} \
            --model-path "computed/models/rnn_metric_learning_${LANGFROM}.pt" \
            --output "computed/tmp/multi_${LANG}_f${LANGFROM}.pkl" \
        ;
    done;
done;

# the order here is super important
./patches/06-join_lists.py \
    --input "computed/tmp/multi_"{en,am,bn,uz,pl,es,sw}".pkl" \
    --output "computed/embd_rnn_metric_learning.pkl" \
;

for LANGFROM in 'en' 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
    ./patches/06-join_lists.py \
        --input "computed/tmp/multi_"{en,am,bn,uz,pl,es,sw}"_f${LANGFROM}.pkl" \
        --output "computed/embd_rnn_metric_learning_f${LANGFROM}.pkl" \
    ;
done;
