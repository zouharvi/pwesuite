#!/usr/bin/bash

mkdir -p data/raw
mkdir -p data/cache

# for LANG in 'am' 'bn' 'uz' 'pl' 'es' 'sw'; do
#   wget "https://data.statmt.org/cc-100/${LANG}.txt.xz" -P data/raw
# done

# wget "https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b" -O "data/raw/cmudict.tmp"
# # the file is not encoded correctly so make sure it's that and invalid codepoints are skipped (-c)
# iconv -c -t utf8 "data/raw/cmudict.tmp" > "data/raw/cmudict-0.7b.txt"
# wget "https://github.com/menelik3/cmudict-ipa/raw/master/cmudict-0.7b-ipa.txt" -O "data/raw/cmudict-0.7b-ipa.txt"

python3 ./create_dataset/preprocess.py
python3 ./create_dataset/add_analogies.py
python3 ./create_dataset/add_human_similarity.py

python3 ./patches/35-tsv_to_csv.py