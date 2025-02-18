import panphon2
import numpy as np
import random
from collections import Counter
from scipy.stats import bernoulli
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, data, p=0.1, k=10, min_occur=5):
        super().__init__()

        self.data = data

        self.ft = panphon2.FeatureTable()
        self.dist = self.ft.feature_edit_distance
        self.length = len(self.data)

        # build vocab
        ipa = set(list(map(lambda x: x[0], self.data)))
        counter = Counter()
        for word in ipa:
            tokens = self.ft.phonemes(word)
            counter.update(tokens)
        
        self.vocab = [w for w, cnt in counter.items() if cnt >= min_occur]
        self.transforms = ['ins', 'del', 'sub']
        self.p = p
        self.k = k # number of random phones to sample to select negative instance
    
    def __len__(self):
        return self.length
        
    def perturb(self, idx):
        word = self.data[idx][0]
        phones = self.ft.phonemes(word)
        perturbed_phones = []
        for i in range(len(phones)):
            if bernoulli.rvs(self.p):
                transform = random.choice(self.transforms)
                if transform == 'ins':
                    tok = random.choice(self.vocab)
                    perturbed_phones.append(phones[i])
                    perturbed_phones.append(tok)
                if transform == 'del':
                    perturbed_phones.append('')
                elif transform == 'sub':
                    tok = random.choice(self.vocab)
                    perturbed_phones.append(tok)
            else:
                perturbed_phones.append(phones[i])
        perturbed_word = "".join(perturbed_phones)

        # sometimes deletion results in an empty string
        if len(perturbed_word) == 0:
            perturbed_word = word

        return (perturbed_word, self.ft.word_to_binary_vectors(perturbed_word))

    def __getitem__(self, idx):
        neg_indices = np.random.randint(0, self.length, size=self.k)

        anchor_ipa = self.data[idx][0]
        negative_ipa = [self.data[neg_idx][0] for neg_idx in neg_indices]
        # negative_ipa = self.data[neg_idx][0]

        anchor_fv = self.data[idx][1]
        negative_fv = [self.data[neg_idx][1] for neg_idx in neg_indices]
        # negative_fv = self.data[neg_idx][1]

        # positive example
        # positive_ipa, positive_fv = self.perturb(idx)
        positive_ipa = anchor_ipa
        positive_fv = anchor_fv


        return {
            'anchor': (anchor_ipa, anchor_fv),
            'positive': (positive_ipa, positive_fv),
            'negative': (negative_ipa, negative_fv)
        }


class PlainDataset(Dataset):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ipa = self.data[idx][0]
        feat_vec = self.data[idx][1]

        return {
            "ipa": ipa,
            "feat_vec": feat_vec
        }