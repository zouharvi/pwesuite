#!/usr/bin/bash

mkdir -p data/raw
mkdir -p data/cache

for LANG in 'am' 'bn' 'uz' 'pl' 'es' 'sw' 'fr' 'de'; do
  wget -nc "https://data.statmt.org/cc-100/${LANG}.txt.xz" -P data/raw
done

wget -nc "https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b" -O "data/raw/cmudict.tmp"
# the file is not encoded correctly so make sure it's that and invalid codepoints are skipped (-c)
iconv -c -t UTF-8 "data/raw/cmudict.tmp" > "data/raw/cmudict-0.7b.txt"
wget -nc "https://github.com/menelik3/cmudict-ipa/raw/master/cmudict-0.7b-ipa.txt" -O "data/raw/cmudict-0.7b-ipa.txt"

# for English epitran
if ! command -v /usr/local/bin/lex_lookup &> /dev/null
then
  cd data/raw
  git clone https://github.com/festvox/flite
  cd flite
  if [ "$(uname)" == "Darwin" ]; then
  sed -i.bak "s/cp \-pd/cp \-pR/g" main/Makefile #this command only needs to run if it is on a MacOS machine.
  fi
  if [[ -z "${NON_ROOT_CUSTOM_INSTALL_DIR}" ]]; then
  echo 'building the normal root install'
  ./configure && make
  sudo make install
  cd testsuite
  make lex_lookup
  sudo cp lex_lookup /usr/local/bin
  else
  echo 'building the non-root workaround'
  ./configure --prefix=${NON_ROOT_CUSTOM_INSTALL_DIR} && make
  make install
  cd testsuite
  make
  echo 'lex_lookup is located in the folder below, add it to path at a location of your choosing' 
  pwd
  fi
  cd ../../../..
fi

wget -nc https://raw.githubusercontent.com/kbatsuren/CogNet/master/CogNet-v0.tsv -O data/raw/CogNet-v0.tsv

python3 ./create_dataset/preprocess.py
python3 ./create_dataset/add_analogies.py
python3 ./create_dataset/add_human_similarity.py
python3 ./create_dataset/add_cognates.py
python3 ./create_dataset/finalize_hf_jsonl.py