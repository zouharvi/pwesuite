import panphon2


def preprocess_dataset(data, features):
    if features == "panphon":
        return preprocess_dataset_panphon(data)
    elif features == "tokenort":
        return preprocess_dataset_token(data, index=0)
    elif features == "tokenipa":
        return preprocess_dataset_token(data, index=1)


def preprocess_dataset_panphon(data):
    f = panphon2.FeatureTable()
    # panphon, token_ipa
    return [(f.word_to_binary_vectors(x[1]), x[1]) for x in data]


def preprocess_dataset_token(data, index):
    from torchtext.vocab import build_vocab_from_iterator
    import torch
    import torch.nn.functional as F

    vocab = build_vocab_from_iterator([x[index] for x in data])

    def token_onehot(characters):
        return F.one_hot(torch.tensor(vocab(list(characters))), num_classes=len(vocab)).float()

    # features, token_ipa
    data = [(token_onehot(x[index]), x[1]) for x in data]

    return data
