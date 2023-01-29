import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from main.utils import reparameterize

class RNN_VAE(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_dim=24,
                 encoder_hidden_dim=128,
                 decoder_input_dim=128,
                 decoder_hidden_dim=128,
                 variational=True
                 ):
        super().__init__()
        self.variational = variational
        # self.layernorm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        ## encoder
        self.encoder = nn.GRU(emb_dim, encoder_hidden_dim, num_layers=1, dropout=0.5, batch_first=True)
        self.to_mu = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)
        self.to_logvar = nn.Linear(encoder_hidden_dim, decoder_hidden_dim)

        ## decoder
        self.embed = nn.Embedding(vocab_size, decoder_input_dim)
        self.decoder = nn.GRU(decoder_input_dim, decoder_hidden_dim, num_layers=1, dropout=0.5, batch_first=True)
        self.output = nn.Linear(decoder_hidden_dim, vocab_size)

    def forward(self, tokens, feature_array):
        # feature_array has shape  (N, L, 24) where L is the phoneme length of the longest word
        # it is essentially a fixed, discrete embedding

        _, encoder_hidden = self.encoder(feature_array, None)
        mu = self.to_mu(encoder_hidden)
        if self.variational:
            logvar = self.to_logvar(encoder_hidden)
            z = reparameterize(mu, logvar)
        else:
            logvar = None
            z = mu

        # TODO: make teacher forcing adjustable
        token_emb = self.embed(tokens)
        decoder_output, _ = self.decoder(token_emb, z)
        decoder_logits = self.output(decoder_output)

        return mu, logvar, decoder_logits


