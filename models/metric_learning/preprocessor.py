import panphon
import panphon2
from main.utils import load_multi_data

def preprocess_dataset(data, features, lang, purpose_key="all"):
    # token_ort, token_ipa, lang, pronunc
    data = [
        (x["token_ort"], x["token_ipa"], x["token_arp"]) for x in load_multi_data(data, purpose_key=purpose_key)
    ]
    data_all = data
    data = [
        x for x in data
        if lang == "all" or x["lang"] == lang
    ]
    if features == "panphon":
        return preprocess_dataset_panphon(data)
    else:
        return preprocess_dataset_token(data_all, data, features)


def preprocess_dataset_panphon(data):
    import numpy as np
    f = panphon2.FeatureTable()
    # panphon, token_ipa
    return [(np.array(f.word_to_binary_vectors(x[1])), x[1]) for x in data]


def preprocess_dataset_token(data_all, data, features):
    from torchtext.vocab import build_vocab_from_iterator
    import torch
    import torch.nn.functional as F

    # TODO: use huggingface and create own vocabulary instead of relying on the one on the disk
    # use the same multi vocabulary across all models
    # a nice side effect is the same number of parameters everywhere
    if features == "tokenort":
        vocab_raw = [c for word in data_all for c in word[0]]
        vocab = build_vocab_from_iterator([[x] for x in vocab_raw])
    elif features == "tokenipa":
        ft = panphon.FeatureTable()
        vocab_raw = [c for word in data_all for c in ft.ipa_segs(word[1])]
        vocab = build_vocab_from_iterator([[x] for x in vocab_raw])

    # TODO: add a default pointer to vocab to point to utils.UNK_SYMBOL

    def token_onehot(word):
        return F.one_hot(torch.tensor(vocab(list(word))), num_classes=len(vocab)).float()

    # features, token_ipa
    if features == "tokenort":
        data = [(token_onehot(x[0]), x[1]) for x in data]
    elif features == "tokenipa":
        ft = panphon.FeatureTable()
        data = [(token_onehot(ft.ipa_segs(x[1])), x[1]) for x in data]

    return data
