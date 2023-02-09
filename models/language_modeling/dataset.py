import panphon
from torch.utils.data import Dataset
from vocab import BOS_IDX, EOS_IDX

class IPATokenDataset(Dataset):
    def __init__(self, input_files, vocab, split_bounds=(0, 1)):
        super().__init__()

        self.ipa_tokens = []
        for fpath in input_files:
            with open(fpath) as f:
                tokens = f.read().split()
            length = len(tokens)
            self.ipa_tokens.extend(tokens[int(length*split_bounds[0]):
                                          int(length*split_bounds[1])])

        self.ft = panphon.FeatureTable()
        self.vocab = vocab

    def __getitem__(self, idx):
        ipa = self.ipa_tokens[idx]
        feature_array = self.ft.word_to_vector_list(ipa, numeric=True)
        tokens = [BOS_IDX] + [self.vocab.get_idx(seg) for seg in self.ft.ipa_segs(ipa)] + [EOS_IDX]

        return {
            'feature_array': feature_array,
            'tokens': tokens,
            'ipa': ipa
        }


    def __len__(self):
        return len(self.ipa_tokens)
