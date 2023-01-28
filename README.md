# Phonetic Representation

Evaluation suite for phonetic (phonological) word embeddings and an additional model based on Panphone distance learning.

The suite contains the following tasks:
- Correlation with Panphone
- Nearest neighbour retrieval with Panphone
- Rhyme detection
- Cognate detection
- Sound analogies
- MT usefulness

Instructions WIP.

Run `pip3 install -e .` to install this repository and its dependencies.

## Embedding evaluation

In order to run all the evaluations, you first need to run the embedding on provided words.
These are found in `data/multi.tsv` (tab-separated values) in the format of:
```
word_ortho	word_ipa	language	pronunciation_information
```
Not all of the informations are provided for all languages, though `word_ipa` is guaranteed to be present.
Specifically, currently only English words contain all these fields.

After running the embedding on each word, save it as either a Pickle or NPZ.
The data structure can be either (1) list of list or numpy arrays or (2) numpy array.
The loader will automatically parse the file and check that the dimensions are consistent.

After this, you are all set to run all the evaluations using `./evaluation/all.py --data your_embd.pkl`.
Alternatively, you can invoke individual tasks (TODO).


<!-- 
Learning a continuous representation for a sequence of discrete vectors of articulatory features.

## Installation
Code was tested with python 3.9
```bash
pip install -r requirements.txt
```

## Training
This is an example configuration to train model
```bash
wandb login <your_credentials> OR wandb disabled
DIM=128
python train.py \
--lang_codes am bn uz pl es sw \
--vocab_file data/vocab_multi.txt \
--batch_size 512 \
--encoder_hidden_dim $DIM \
--decoder_hidden_dim $DIM \
--decoder_input_dim $DIM \
--lr 0.0001  \
--kl_mult 10 \
--wandb_name example_run
```

## Inference
Once a model is trained, run the following code to save the predicted continuous representation into an `npy` file.
```bash
python inference.py \
--input_path  ./data/inference_example.txt \
--output_path ./predictions/test_inference.npy \
--model_path  ./checkpoints/<your_model>.pt
``` -->