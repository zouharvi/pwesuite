#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/phonetic-representation/

# rsync -azP data/multi.tsv euler:/cluster/work/sachan/vilem/phonetic-representation/data/multi.tsv
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_rnn_metric_learning/{panphon,tokenipa}.pkl computed/embd_rnn_metric_learning/
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_baseline/ computed/embd_baseline/
