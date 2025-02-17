#!/usr/bin/bash

mkdir -p data/raw
mkdir -p data/cache

for LANG in 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
  wget -nc "https://data.statmt.org/cc-100/${LANG}.txt.xz" -P data/raw
done

wget -nc "https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b" -O "data/raw/cmudict.tmp"
# th;e file is not encoded correctly so make sure it's that and invalid codepoints are skipped (-c)
iconv -c -t UTF-8 "data/raw/cmudict.tmp" > "data/raw/cmudict-0.7b.txt"
wget -nc "https://github.com/menelik3/cmudict-ipa/raw/master/cmudict-0.7b-ipa.txt" -O "data/raw/cmudict-0.7b-ipa.txt"

# for English epitran
if ! command -v /usr/local/bin/lex_lookup &> /dev/null
then
  wget http://tts.speech.cs.cmu.edu/awb/flite-2.0.5-current.tar.bz2 -O data/raw/flite-2.0.5-current.tar.bz2
  cd data/raw
  tar xjf flite-2.0.5-current.tar.bz2
  cd flite-2.0.5-current
  if [ "$(uname)" == "Darwin" ]; then
  sed -i.bak "s/cp \-pd/cp \-pR/g" main/Makefile #this command only needs to run if it is on a MacOS machine.
  fi
  ./configure && make
  sudo make install
  cd testsuite
  make lex_lookup
  sudo cp lex_lookup /usr/local/bin
  cd ../../../..
fi

# wget -nc https://raw.githubusercontent.com/kbatsuren/CogNet/master/CogNet-v0.tsv -O data/raw/CogNet-v0.tsv

python3 ./create_dataset/preprocess.py
python3 ./create_dataset/add_analogies.py
python3 ./create_dataset/add_human_similarity.py
python3 ./create_dataset/add_cognates.py

python3 ./create_dataset/finalize_hf_jsonl.py