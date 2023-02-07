#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/phonetic-representation/

# rsync -azP data/multi.tsv euler:/cluster/work/sachan/vilem/phonetic-representation/data/multi.tsv
# rsync -azP data/cache/cognates.pkl euler:/cluster/work/sachan/vilem/phonetic-representation/data/cache/
# rsync -azP data/human_similarity.csv euler:/cluster/work/sachan/vilem/phonetic-representation/data/
# rsync -azP data/cache/ euler:/cluster/work/sachan/vilem/phonetic-representation/data/cache/
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_baseline/bert.pkl computed/embd_baseline/bert.pkl
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/logs/eval_all_rnn_panphon_d*.log logs/
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_rnn_metric_learning/{panphon,tokenipa,tokenort}.pkl computed/embd_rnn_metric_learning/
# rsync -azP euler:/cluster/work/sachan/vilem/phonetic-representation/computed/embd_other/sharma.pkl computed/embd_other/