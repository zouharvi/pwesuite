# Phonetic Word Embeddings Suite (`PWESuite`)

Evaluation suite for phonetic (phonological) word embeddings and an additional model based on Panphone distance learning.

The suite contains the following tasks:
- Correlation with human sound similarity judgement
- Correlation with articulatory distance
- Nearest neighbour retrieval
- Rhyme detection
- Cognate detection
- Sound analogies

Run `pip3 install -e .` to install this repository and its dependencies.

## Embedding evaluation

In order to run all the evaluations, you first need to run the embedding on provided words.
These can be downloaded from [our Huggingface dataset](https://huggingface.co/datasets/zouharvi/pwesuite-eval):
```
>>> from datasets import load_dataset
>>> dataset = load_dataset("zouharvi/pwesuite-eval")
>>> dataset["train"][10]
{'token_ort': 'aachener', 'token_ipa': 'ɑːkən', 'lang': 'en', 'purpose': 'main', 'token_arp': 'AA1 K AH0 N ER0'}
```
Note that each line contains `token_ort`, `token_ipa`, `token_arp` and `lang`.
For training, only the words marked with `purpose=="main"` should be used.
Note that unknown/low frequency phonemes or letters are replaced with `😕`.
You can also generate the `data/multi.csv` file locally by running `main/prepare_data.sh`.

After running the embedding **for each line/word**, save it as either a Pickle or NPZ. 
The data structure can be either (1) list of list or numpy arrays or (2) numpy array.
The loader will automatically parse the file and check that the dimensions are consistent.

After this, you are all set to run all the evaluations using `./suite_evaluation/eval_all.py --embd your_embd.pkl`.
Alternatively, you can invoke individual tasks: `./suite_evaluation/eval_{correlations,human_similarity,retrieval,analogy,rhyme,cognate}.py`.


## Misc

Contact the authors if you encounter any issues using this evaluation suite.
Read the [associated paper](https://arxiv.org/abs/2304.02541) and for now cite as:

```
@article{zouhar2023pwesuite,
  title={{PWESuite}: {P}honetic Word Embeddings and Tasks They Facilitate},
  author={Zouhar, Vil{\'e}m and Chang, Kalvin and Cui, Chenxuan and Carlson, Nathaniel and Robinson, Nathaniel and Sachan, Mrinmaya and Mortensen, David},
  journal={arXiv preprint arXiv:2304.02541},
  year={2023},
  url={https://arxiv.org/abs/2304.02541}
}
```