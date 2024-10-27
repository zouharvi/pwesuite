# Phonetic Word Embeddings Suite (`PWESuite`)

Evaluation suite for phonetic (phonological) word embeddings and an additional model based on Panphone distance learning.
This repository accompanies the paper [PWESuite: Phonetic Word Embeddings and Tasks They Facilitate](https://arxiv.org/abs/2304.02541) at LREC-COLING 2024.
Watch [12-minute introduction to PWESuite](https://www.youtube.com/watch?v=XJ9bAPaJlyc).

> **Abstract:** Mapping words into a fixed-dimensional vector space is the backbone of modern NLP. While most word embedding methods successfully encode semantic information, they overlook phonetic information that is crucial for many tasks. We develop three methods that use articulatory features to build phonetically informed word embeddings. To address the inconsistent evaluation of existing phonetic word embedding methods, we also contribute a task suite to fairly evaluate past, current, and future methods. We evaluate both (1) intrinsic aspects of phonetic word embeddings, such as word retrieval and correlation with sound similarity, and (2) extrinsic performance on tasks such as rhyme and cognate detection and sound analogies. We hope our task suite will promote reproducibility and inspire future phonetic embedding research.



[<img src="https://img.youtube.com/vi/XJ9bAPaJlyc/maxresdefault.jpg" width=600px>](https://www.youtube.com/watch?v=XJ9bAPaJlyc)

<!--p align="center">
  <img src="https://github.com/zouharvi/pwesuite/assets/7661193/e8db7af0-cccf-425a-8a3c-4f260d5abab7" width="500em">
</p-->

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
>>> dataset = load_dataset("zouharvi/pwesuite-eval", split="train")
>>> dataset[10]
{'token_ort': 'aachener', 'token_ipa': 'ɑːkən', 'lang': 'en', 'purpose': 'main', 'token_arp': 'AA1 K AH0 N ER0'}
```
Note that each line contains `token_ort`, `token_ipa`, `token_arp` and `lang`.
For training, only the words marked with `purpose=="main"` should be used.
Note that unknown/low frequency phonemes or letters are replaced with `😕`.

After running the embedding **for each line/word**, save it as either a Pickle or NPZ. 
The data structure can be either (1) list of list or numpy arrays or (2) numpy array.
The loader will automatically parse the file and check that the dimensions are consistent.

After this, you are all set to run all the evaluations using `./suite_evaluation/eval_all.py --embd your_embd.pkl`.
Alternatively, you can invoke individual tasks: `./suite_evaluation/eval_{correlations,human_similarity,retrieval,analogy,rhyme,cognate}.py`.

For a demo, see [this Jupyter notebook](demo.ipynb).

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

## Compute details

The most compute-intensive tasks were training the Metric Learner and Triplet Margin, which took 1/4 and 2 hours on GTX 1080 Ti, respectively.
For the research presented in this paper, we estimate 100 GPU hours overall.

The BERT embeddings were extracted as an average across the last layer.
The INSTRUCTOR embeddings were used with the prompt _"Represent the word for sound similarity retrieval:"_.
For BPEmb and fastText, we used the best models (highest training data) and dimensionality of 300.

The metric learner uses bidirectional LSTM with 2 layers, hidden state size of 150 and dropout of 30%.
The batch size is 128 and the learning rate is 0.01.
The autoencoder follows the same hyperparameters both for the encoder and decoder.
The difference is its learning size, 0.005, which was chosen empirically.


![poster](https://github.com/zouharvi/pwesuite/assets/7661193/e2539886-30b1-4fbd-b768-ec3a61dfa1ce)
