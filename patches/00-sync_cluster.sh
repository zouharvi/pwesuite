#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/phonetic-representation/

# rsync -azP data/multi.tsv euler:/cluster/work/sachan/vilem/phonetic-representation/data/multi.tsv
# rsync -azP data/cache/ euler:/cluster/work/sachan/vilem/phonetic-representation/data/cache/
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_baseline/ computed/embd_baseline/
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_rnn_metric_learning/{panphon,tokenipa,tokenort}.pkl computed/embd_rnn_metric_learning/