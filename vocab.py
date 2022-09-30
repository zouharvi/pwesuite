UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<bos>', '<eos>']


class Vocab:
    def __init__(self, tokens=None, tokens_file=None):
        if tokens_file is not None:
            with open(tokens_file) as f:
                tokens = f.read().split()

        self.v2i = {special_token: idx for idx, special_token in enumerate(SPECIAL_TOKENS)}

        for idx, token in enumerate(set(tokens)):
            self.v2i[token] = idx + len(SPECIAL_TOKENS)

        self.i2v = {v: k for k, v in self.v2i.items()}
        assert len(self.v2i) == len(self.i2v)

    def to_string(self, index_sequence, remove_special=True, return_list=False):
        '''
        * returns string representation of index sequence
        '''
        ret = []
        for idx in index_sequence:
            idx = idx.item()
            if remove_special:
                if idx in {UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX}:
                    continue
            ret.append(self.i2v[idx])
        if return_list:
            return ret
        else:
            return ''.join(ret)

    def get_idx(self, v):
        return self.v2i.get(v, UNK_IDX)

    def __len__(self):
        return len(self.v2i)

    def __getitem__(self, idx):
        return self.i2v.get(idx, self.i2v[UNK_IDX])

    def __iter__(self):
        for idx, tkn in sorted(self.i2v.items(), key=lambda x: x[0]):
            yield idx,