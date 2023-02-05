# Imports are done within functions so that they are not needlessly loaded when a different function is used

LANGS = ['en', 'am', 'bn', 'uz', 'pl', 'es', 'sw']
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

# TODO: use huggingface
def load_multi_data(path="data/multi.tsv", purpose_key="main"):
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

def load_embd_data(path):
    if path.endswith(".pkl") or path.endswith(".pickle"):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif path.endswith(".npz"):
        import numpy as np
        with open(path, "rb") as f:
            data = np.load(f)
    else:
        raise Exception("Unknown file suffix: " + path)

    return data


def get_device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"