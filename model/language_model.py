import torch
from torch import nn
from vocab import BOS_IDX, EOS_IDX


class AutoregressiveLM(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_layers, dropout, bidirectional, vocab_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.model = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(
            in_features=2 * hidden_dim if bidirectional else hidden_dim,
            out_features=vocab_size
        )
        # only for <bos>. <eos> will never be fed as input
        self.bos_embedding = nn.Embedding(1, feature_size)

    def forward(self, segment_features, hidden, cell):
        '''
        Inputs:
            token - (B,) token indices for one index in the sequence, across all batches
            segment_features - (B, 1, 24) one token's panphon features from each batch
            hidden - previous token's hidden state
            cell - previous token's cell state

        Returns:
            logits - (B, S, V)
            hidden - (2 * num_layers, hidden_dim) or (num_layers, hidden_dim) if bidirectional=False
            cell - (2 * num_layers, hidden_dim) or (num_layers, hidden_dim) if bidirectional=False

        Where
            B: batch size
            S: sequence length
            V: vocab size
        '''
        # for one segment, pass its panphon features through the LSTM
        # then pass through the linear layer to predict the next token
        outputs, (hidden, cell) = self.model(segment_features, (hidden, cell))
        logits = self.linear(outputs)

        return logits, (hidden, cell)

    def pool(self, feature_sequence):
        '''
        Inputs:
            feature_sequence - (B, S + 1, 24) - <BOS> embedding should be in the first index

        Returns:
            Word-level embedding for the sequence of features (feature matrix) - (B, 2 * E) or (B, E) if bidirectional=False
        '''
        outputs, (_, _) = self.model(feature_sequence)

        # take the last token
        # TODO: is this really a pooled representation of the sequence when we are training it to predict EOS???
        # TODO: isn't taking the last token a problem when you have batches of different lengths?
        #   this will correspond to <PAD> for sequences shorter than the longest seq in the batch
        # TODO: within each batch, take the representation of the token without padding
        return outputs[:, -1, :]
