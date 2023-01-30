#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/phonetic-representation/

# rsync -azP data/multi.tsv euler:/cluster/work/sachan/vilem/phonetic-representation/data/multi.tsv
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_rnn_metric_learning/ computed/embd_rnn_metric_learning/
