# Phonetic Word Embeddings Suite `PWESuite`

[![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](https://aclanthology.org/2024.lrec-main.1168/)
&nbsp;
[![YouTube video](https://img.shields.io/badge/🎥%20YouTube%20video-F00.svg)](https://www.youtube.com/watch?v=XJ9bAPaJlyc)

Evaluation suite for phonetic (phonological) word embeddings and an additional model based on Panphone distance learning.
This repository accompanies the paper [PWESuite: Phonetic Word Embeddings and Tasks They Facilitate](https://aclanthology.org/2024.lrec-main.1168/) at LREC-COLING 2024.
Watch [12-minute introduction to PWESuite](https://www.youtube.com/watch?v=XJ9bAPaJlyc).

> **Abstract:** Mapping words into a fixed-dimensional vector space is the backbone of modern NLP. While most word embedding methods successfully encode semantic information, they overlook phonetic information that is crucial for many tasks. We develop three methods that use articulatory features to build phonetically informed word embeddings. To address the inconsistent evaluation of existing phonetic word embedding methods, we also contribute a task suite to fairly evaluate past, current, and future methods. We evaluate both (1) intrinsic aspects of phonetic word embeddings, such as word retrieval and correlation with sound similarity, and (2) extrinsic performance on tasks such as rhyme and cognate detection and sound analogies. We hope our task suite will promote reproducibility and inspire future phonetic embedding research.


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
@inproceedings{zouhar-etal-2024-pwesuite,
    title = "{PWES}uite: Phonetic Word Embeddings and Tasks They Facilitate",
    author = "Zouhar, Vil{\'e}m  and
      Chang, Kalvin  and
      Cui, Chenxuan  and
      Carlson, Nate B.  and
      Robinson, Nathaniel Romney  and
      Sachan, Mrinmaya  and
      Mortensen, David R.",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1168/",
    pages = "13344--13355",
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

## Development
### Notes
- This project has dependencies which determine usable Python 3 versions, if you don't have admin rights on your system consider configuring this in a docker container.
- This project requires local admin or sudo privelges depending on your instituion's priveleges policy and so to install epitran you may need to follow a workaround. This is detailed below.
- This project is setup with relative directories to maintain portability, to set the top level directory so imports work you will need to export pwesuite's location to path. This is detailed below.
- This code is written to work on Linux or MacOS only.
- Creating your own copy of the data may not be nessecary as you can instead download the data from huggingface.


### Development setup

If you are not a developer/contributor already you will have to fork and raise a PR rather than a branch. So follow all steps after the git clone to get setup.

```
> git clone git@github.com:zouharvi/pwesuite.git
> cd pwesuite 
> git branch crazy-feature
> python -m venv dev-venv
> source dev-venv/bin/activate
> pip install -e .
Congratulations you are now ready to develop in PWESuite.
```
### Creating your own copy of the data (with sudo)

If you have sudo privileges this will also configure and install cmu flite as lex_lookup is a required tool. 

```
> cd pwesuite
> ./create_dataset/all.sh
```

### Creating your own copy of the data (without sudo)

If you are on a system without sudo and do not want to build your own Docker container this is a workaround.

```
> cd pwesuite
> export NON_ROOT_CUSTOM_INSTALL_DIR=/YOUR_PATH_TO_DIR_HERE
> ./create_dataset/all.sh
#! Helpful prompt 
#! Now you will have a path to the binary for lex_lookup so you need to add it to path
> export PATH=/YOUR_PATH_TO_LEX_LOOKUP/lex_lookup:$PATH

```
Please note your export (i.e environment variable) will only be set for the lifetime of the terminal/shell you are in. For all.sh this is not an issue because you only need to set the custom directory while the code is compiled. It would pay to export the path to the lex_lookup binary every time you run anything to do with PWEsuite.

The final export to add lex_lookup to path preprends it so it will be encountered first.