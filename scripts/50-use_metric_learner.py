"""
Trained as 

FEATURES="token_ort"
LANG="all"
python3 \
    ./models/metric_learning/train.py \
        --lang ${LANG} \
        --save-model-path \"computed/models/rnn_metric_learning_${FEATURES}_${LANG}.pt\" \
        --number-thousands 10000 \
        --target-metric \"l2\" \
        --features ${FEATURES} \
        --epochs 20 \
    ;
"""