import panphon2
from main.utils import load_multi_data

def preprocess_dataset(features, lang, purpose_key="all"):
    # token_ort, token_ipa, lang, pronunc
    data_all = [
        x for x in load_multi_data(purpose_key=purpose_key)
    ]
    data = [
        (x["token_ort"], x["token_ipa"], x["token_arp"]) for x in data_all
        if lang == "all" or x["lang"] == lang
    ]
    data_all = [
        (x["token_ort"], x["token_ipa"], x["token_arp"]) for x in data_all
    ]
    print("Loaded", len(data))
    print("Loaded (all)", len(data_all))
    if features == "panphon":
        return preprocess_dataset_panphon(data)
    else:
        return preprocess_dataset_token(data_all, data, features)


def preprocess_dataset_panphon(data):
    import numpy as np
    f = panphon2.FeatureTable()
    # panphon, token_ipa
    return [(np.array(f.word_to_binary_vectors(x[1])), x[1]) for x in data]


def build_vocab(data):
    import main.utils
    # force-add UNK symbol
    characters = {main.utils.UNK_SYMBOL} | set(data)
    vocab = {char: idx for idx, char in enumerate(characters)}
    return vocab


def preprocess_dataset_token(data_all, data, features):
    import torch
    import torch.nn.functional as F
    import panphon

    # The vocab is based on data_all, so as long as the upstream dataset doesn't change, the vocab will be the same

    if features == "tokenort":
        vocab_raw = [c for word in data_all for c in word[0]]
    elif features == "tokenipa":
        ft = panphon.FeatureTable()
        vocab_raw = [c for word in data_all for c in ft.ipa_segs(word[1])]
    else:
        raise ValueError("Unsupported feature type")
    
    vocab = build_vocab(vocab_raw)
    print("Vocabulary size", len(vocab))

    def token_onehot(word):
        indices = [vocab[c] for c in word if c in vocab]
        return F.one_hot(torch.tensor(indices), num_classes=len(vocab)).float()
    
    if features == "tokenort":
        data = [(token_onehot(x[0]), x[1]) for x in data]
    elif features == "tokenipa":
        ft = panphon.FeatureTable()
        data = [(token_onehot(ft.ipa_segs(x[1])), x[1]) for x in data]
    
    return data
