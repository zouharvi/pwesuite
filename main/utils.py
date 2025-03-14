# Imports are done within functions so that they are not needlessly loaded when a different function is used

LANGS = ['en', 'am', 'bn', 'uz', 'pl', 'es', 'sw', 'fr', 'de']
UNK_SYMBOL = "😕"


def collate_fn(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from models.vae_old.vocab import PAD_IDX

    feature_array = [torch.tensor(b['feature_array']) for b in batch]
    tokens = [torch.tensor(b['tokens']) for b in batch]
    feature_array = pad_sequence(
        feature_array, padding_value=PAD_IDX, batch_first=True)
    tokens = pad_sequence(tokens, padding_value=PAD_IDX, batch_first=True)

    return {
        'feature_array': feature_array.float(),
        'tokens': tokens,
    }

def load_multi_data_raw(path="data/multi.tsv", purpose_key="main"):
    print("Loading data")
    data = [
        l.rstrip("\n").split("\t")
        for l in open(path, "r")
        if len(l) > 1
    ]
    if purpose_key == "all":
        return data
    else:
        data = [
            x
            for x in data
            if x[3] == purpose_key
        ]
    return data


def load_analogies_data(lang):
    print("Loading analogies data")
    import datasets
    import collections
    data = datasets.load_dataset("zouharvi/pwesuite-eval", split="train")
    data = [x for x in data if x["lang"] == lang and x["purpose"] == "analogy"]

    data_analogies = collections.defaultdict(lambda: ["", "", "", ""])
    for x in data:
        analogy_index, word_index = x["extra_index"].split("_")
        data_analogies[int(analogy_index)][int(word_index)] = (x["token_ort"], x["token_ipa"], x["token_arp"])
    return list(data_analogies.values())


def load_cognates_data():
    print("Loading cognates data")
    import datasets
    import collections
    data = datasets.load_dataset("zouharvi/pwesuite-eval", split="train")
    data = [x for x in data if x["lang"] == "multi" and x["purpose"] == "cognate"]

    data_cognates = collections.defaultdict(lambda: ["", "", ""])
    for x in data:
        analogy_index, word_index = x["extra_index"].split("_")
        data_cognates[int(analogy_index)][int(word_index)] = (x["token_ort"], x["token_ipa"], x["token_arp"])
    return list(data_cognates.values())

def load_multi_data(purpose_key="main"):
    print("Loading multi data")
    import datasets
    data = datasets.load_dataset("zouharvi/pwesuite-eval", split="train")

    if purpose_key == "all":
        return list(data)
    else:
        data = [
            x
            for x in data
            if x["purpose"] == purpose_key
        ]
    return data

def load_embd_data(path):
    if path.endswith(".pkl") or path.endswith(".pickle"):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".npz") or path.endswith(".npy"):
        import numpy as np
        data = np.load(path, allow_pickle=True)
    else:
        raise Exception("Unknown file suffix: " + path)

    return data


def get_device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"