import panphon2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
import random
import tqdm
import numpy as np
from models.metric_learning.intrinsic_evaluator import Evaluator
from main.utils import get_device

DEVICE = get_device()

class RNNAutoencoder(torch.nn.Module):
    def __init__(
        self,
        feature_size=24,
        dimension=300,
        safe_eval=False,
    ):
        super().__init__()

        self.model_encoder = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=dimension//2,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.model_decoder = torch.nn.LSTM(
            input_size=dimension,
            hidden_size=dimension//2,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.decoder_projection = torch.nn.Linear(dimension, feature_size)
        self.batch_size_eval = 2048
        self.batch_size_train = 128

        self.loss = torch.nn.CrossEntropyLoss()
        self.panphon_distance = panphon2.FeatureTable().feature_edit_distance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)

        self.evaluator = Evaluator(safe_eval=safe_eval)

        # move the model to GPU
        self.to(DEVICE)

    def evaluate_intrinsic(self, data, key=None):
        print("Evaluating everything for", key)

        self.eval()
        data_loader = DataLoader(
            data, batch_size=self.batch_size_eval, shuffle=False,
            collate_fn=lambda x: [y[0] for y in x]
        )

        # compute embeddings in batches on GPU
        data_embd = [
            self.forward(x).cpu().detach().numpy()
            for x in data_loader
        ]

        # flatten
        data_embd = [x for y in data_embd for x in y]
        data_sims = cosine_distances(data_embd)
        # data_sims = np.ravel(data_sims)

        pearson_corr, spearman_corr = self.evaluator.evaluate_corr(
            data, data_sims, key=key
        )

        retrieval_rank = self.evaluator.evaluate_retrieval(
            data, data_sims, key=key
        )
        out_str = f"ρp={pearson_corr:.0%} ρs={spearman_corr:.0%} rank={retrieval_rank:.0f}"
        return out_str, retrieval_rank, (pearson_corr, spearman_corr)
        

    def evaluate_dev_loss(self, data_dev):
        self.eval()
        losses = []

        data_loader = DataLoader(
            list(data_dev), batch_size=self.batch_size_train, shuffle=True,
            collate_fn=lambda x: (
                # tok_features
                [y[0] for y in x],
                # tok_ipa
                [y[1] for y in x],
            )
        )

        for (tok1_features, tok1_ipa) in tqdm.tqdm(data_loader):
            loss = self.train_step(tok1_features)
            losses.append(loss.cpu().detach())

        return np.average(losses)

    def forward(self, ws):
        # TODO: here we use -1 for padding because 0.0 is already
        # used somewhere. This may not matter much but good to be aware of.
        ws = torch.nn.utils.rnn.pad_sequence(
            [torch.Tensor(x) for x in ws],
            batch_first=True, padding_value=-1.0,
        ).to(DEVICE)
        output, (h_n, c_n) = self.model_encoder(ws)

        # take last vector for all elements in the batch
        output = output[:, -1, :]

        return output

    def decode(self, w1embd, ws1):
        prev_out = w1embd
        prev_hidden = None
        batch_loss = 0

        if type(ws1[0]) is torch.Tensor:
            ws1 = [x.numpy() for x in ws1]

        # pad array to same size
        max_len = max([len(x) for x in ws1])
        pad_value = [1]*len(ws1[0][0])
        padding = [[pad_value]*(max_len-len(x)) for x in ws1]
        ws1padded = torch.tensor(np.array([list(x) + y for x,y in zip(ws1, padding)])).float().to(DEVICE)
        ws_lens = torch.tensor([len(x) for x in ws1])

        # TODO normalize?

        for t in range(max_len):
            prev_out, prev_hidden = self.model_decoder(prev_out, prev_hidden)

            prediction = self.decoder_projection(prev_out)

            # super cool masking and paralelization
            mask = t < ws_lens
            prediction = prediction[mask]
            true_class = ws1padded[mask,t]
            batch_loss += self.loss(prediction, true_class)
        
            # old unparalelized version
            # for sample_i, sample in enumerate(prediction):
            #     # no point decoding from a word that is done
            #     if t >= len(ws1[sample_i]):
            #         continue
            #     true_class = torch.from_numpy(ws1[sample_i][t]).float().to(DEVICE)
            #     # normalize to 1 (only matters for multi-class of panphon)
            #     true_class /= true_class.sum()
            #     batch_loss += self.loss(sample, true_class)

        return batch_loss

    def train_step(self, ws1):
        w1embd = self.forward(ws1)
        batch_loss = self.decode(w1embd, ws1)

        return batch_loss

    def train_epochs(
        self, data_train, data_dev,
        epochs=1000, eval_train_full=False,
    ):
        # set maximum prev_eval_dev_rank
        prev_eval_dev_rank = len(data_dev)

        for epoch in range(epochs):
            self.train()

            data_loader = DataLoader(
                list(data_train), batch_size=self.batch_size_train, shuffle=True,
                collate_fn=lambda x: (
                    # tok_features
                    [y[0] for y in x],
                    # tok_ipa
                    [y[1] for y in x],
                )
            )

            losses = []

            # it's singular names here but actually is full batches
            for (tok1_features, tok1_ipa) in tqdm.tqdm(data_loader):
                loss = self.train_step(tok1_features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.cpu().detach())
                
            dev_loss_avg = self.evaluate_dev_loss(data_dev)

            print(f"Epoch {epoch}")
            print(
                f"Train loss {np.average(losses):8.5f}",
                f"Dev loss {dev_loss_avg:8.5f}",
            )

            if epoch % 3 == 0:
                eval_dev, eval_dev_rank, (eval_dev_pearson, eval_dev_spearman) = self.evaluate_intrinsic(
                    # use only 500 to speed things up
                    data_dev[:500], key="dev"
                )
                # use only part of the training data for evaluation unless specified otherwise
                eval_train, _, _ = self.evaluate_intrinsic(
                    data_train if eval_train_full else data_train[:500], key="train"
                )

                print("Train", eval_train, "|||", "dev", eval_dev)

                # quit when we get worse
                if eval_dev_rank > prev_eval_dev_rank:
                    return
                prev_eval_dev_rank = eval_dev_rank