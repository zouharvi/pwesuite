# Phonetic Word Embeddings Suite `PWESuite`

[![Paper](https://img.shields.io/badge/📜%20paper-481.svg)](https://aclanthology.org/2024.lrec-main.1168/)
&nbsp;
[![YouTube video](https://img.shields.io/badge/🎥%20YouTube%20video-F00.svg)](https://www.youtube.com/watch?v=XJ9bAPaJlyc)
&nbsp;
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FCD21D)](https://huggingface.co/collections/zouharvi/pwesuite-67b6e860a18e514d36293e74)

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
Pre-trained modelse are available [here on Huggingface](https://huggingface.co/zouharvi/PWESuite-metric_learner).

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

## Using embeddings

See the instructions on HuggingFace models. For example, the `rnn_metric_learning_token_orth` model can be loaded and used as:
```
from models.metric_learning.model import RNNMetricLearner
from models.metric_learning.preprocessor import preprocess_dataset_foreign
from main.utils import load_multi_data
import torch
import tqdm
import math

data = load_multi_data(purpose_key="all")
data = preprocess_dataset_foreign(
  [
    {"token_ort": "Hello", "token_ipa": None},
    {"token_ort": "what", "token_ipa": None},
    {"token_ort": "is", "token_ipa": None},
    {"token_ort": "pwesuite", "token_ipa": None},
  ],
  features="token_ort"
)

model = RNNMetricLearner(
    dimension=300,
    feature_size=data[0][0].shape[1],
)
model.load_state_dict(torch.load("computed/models/rnn_metric_learning_token_orth_all.pt"))

# some cheap paralelization
BATCH_SIZE = 32
data_out = []
for i in tqdm.tqdm(range(math.ceil(len(data) / BATCH_SIZE))):
    batch = [f for f, _ in data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]]
    data_out += list(
        model.forward(batch).detach().cpu().numpy()
    )

assert len(data) == len(data_out)
assert all([len(x) == 300 for x in data_out])
```

## Misc

Contact the authors if you encounter any issues using this evaluation suite.
Read the [associated paper](https://aclanthology.org/2024.lrec-main.1168/) and for now cite as:

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

Because of a dependency on panphon2, this project can use Python up to only 3.11, see [the tracking issue](https://github.com/zouharvi/pwesuite/issues/15).

### Development setup

If you are not a developer/contributor already you will have to fork and raise a PR rather than a branch. So follow all steps after the git clone to get setup.

```
> git clone git@github.com:zouharvi/pwesuite.git
> cd pwesuite 
> git branch crazy-feature
> python -m venv dev-venv
> source dev-venv/bin/activate
> pip install -e .
```
Congratulations you are now ready to develop in PWESuite.

### Contributors

We sincerely thank [@JeffBezos64](https://github.com/JeffBezos64) for contributing to this project.

### Replicating dataset

Creating your own copy of the data is not strictly necessary, as you can instead download the data from huggingface.
If you wish to replicate the data locally, having sudo privileges might be necessary as we need to install `flite`.
The following script should handle everything for you in this case (on Linux and MacOS):

```bash
./create_dataset/all.sh
```
