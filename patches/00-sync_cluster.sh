#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/pwesuite/

# rsync -azP euler:/cluster/work/sachan/vilem/pwesuite/data/multi.tsv data/multi.tsv
# rsync -azP data/multi.tsv euler:/cluster/work/sachan/vilem/pwesuite/data/multi.tsv
# rsync -azP data/cache/cognates.pkl euler:/cluster/work/sachan/vilem/pwesuite/data/cache/
# rsync -azP data/human_similarity.csv euler:/cluster/work/sachan/vilem/pwesuite/data/
# rsync -azP data/cache/ euler:/cluster/work/sachan/vilem/pwesuite/data/cache/
# rsync -azP euler:/cluster/work/sachan/vilem/pwesuite/computed/embd_baseline/bert.pkl computed/embd_baseline/bert.pkl
# rsync -azP euler:/cluster/work/sachan/vilem/pwesuite/logs/eval_all_rnn_panphon_d*.log logs/
# rsync -azP euler:/cluster/work/sachan/vilem/pwesuite/logs/eval_mismatch_rnn_tokenipa.log computed/mismatch_tokenipa.log
# rsync -azP euler:/cluster/work/sachan/vilem/pwesuite/computed/embd_rnn_metric_learning/{panphon,tokenipa,tokenort}.pkl computed/embd_rnn_metric_learning/
# rsync -azP euler:/cluster/work/sachan/vilem/pwesuite/computed/embd_other/sharma.pkl computed/embd_other/