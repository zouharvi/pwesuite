import panphon
import panphon2
from main.utils import load_multi_data

def preprocess_dataset(features, lang, purpose_key="all"):
    # token_ort, token_ipa, lang, pronunc
    data = load_multi_data(purpose_key=purpose_key)
    data = [
        x for x in data
        if lang == "all" or x["lang"] == lang
    ]
    print("Loaded", len(data))
    return preprocess_dataset_foreign(data, features)
    

def preprocess_dataset_foreign(data, features):
    import torch
    import torch.nn.functional as F
    
    def token_onehot(word):
        indices = [vocab[c] for c in word if c in vocab]
        return F.one_hot(torch.tensor(indices), num_classes=len(vocab)).float()
    
    if features == "token_ort":
        vocab = get_vocab_all("token_ort")
        print("Vocabulary size", len(vocab))
        # feature, token_ipa
        return [(token_onehot(x["token_ort"]), x["token_ipa"]) for x in data]
    elif features == "token_ipa":
        vocab = get_vocab_all("token_ipa")
        print("Vocabulary size", len(vocab))
        ft = panphon.FeatureTable()
        # feature, token_ipa
        return [(token_onehot(ft.ipa_segs(x["token_ipa"])), x["token_ipa"]) for x in data]
    elif features == "panphon":
        import numpy as np
        f = panphon2.FeatureTable()
        # panphon, token_ipa
        return [(np.array(f.word_to_binary_vectors(x["token_ipa"])), x["token_ipa"]) for x in data]
    else:
        raise ValueError("Unsupported feature type")


def get_vocab_all(features):
    """
    The vocab is based on data_all, so as long as the upstream dataset doesn't change, the vocab will be the same
    """

    import panphon
    import main.utils
    
    data_all = load_multi_data(purpose_key="all")
    if features == "token_ort":
        vocab_raw = [c for word in data_all for c in word["token_ort"]]
    elif features == "token_ipa":
        ft = panphon.FeatureTable()
        vocab_raw = [c for word in data_all for c in ft.ipa_segs(word["token_ipa"])]
    else:
        raise ValueError("Unsupported feature type")
    
    # force-add UNK symbol
    characters = {main.utils.UNK_SYMBOL} | set(vocab_raw)
    vocab = {char: idx for idx, char in enumerate(characters)}
    return vocab
