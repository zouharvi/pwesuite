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
--vocab_file data/vocab_am_bn_uz_pl_es_sw.txt \
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