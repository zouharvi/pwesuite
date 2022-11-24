import panphon2
import torch
from torch.utils.data import DataLoader
import random
import tqdm
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances
import multiprocess as mp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RNNMetricLearner(torch.nn.Module):
    def __init__(
        self,
        target_metric,
        feature_size=24, panphon_vectors=True,
    ):
        super().__init__()

        self.model = torch.nn.LSTM(
            input_size=feature_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        # TODO: use characters instead of vectors but maybe that doesn't matter
        self.panphon_vectors = panphon_vectors

        # TODO: contrastive learning
        self.loss = torch.nn.MSELoss()
        self.panphon_distance = panphon2.FeatureTable().feature_edit_distance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-3)

        if target_metric == "l2":
            self.dist_embd = torch.nn.PairwiseDistance(p=2)
        elif target_metric == "ip":
            self.dist_embd = torch.nn.CosineSimilarity()
        else:
            raise Exception(f"Unknown metric {target_metric}")

        # move the model to GPU
        self.to(DEVICE)

        self.panphon_dist_cache = {}

    def forward(self, ws):
        # TODO: here we use -1 for padding because 0.0 is already
        # used somewhere. This may not matter much but good to be aware of.
        ws = torch.nn.utils.rnn.pad_sequence(
            [torch.Tensor(x) for x in ws],
            batch_first=True, padding_value=-1.0,
        ).to(DEVICE)
        output, (h_n, c_n) = self.model(ws)

        # take last vector for all elements in the batch
        output = output[:, -1, :]

        return output

    def train_step(self, ws1, ws2, dist_true):
        w1embd = self.forward(ws1)
        w2embd = self.forward(ws2)

        # compute distance in embd space
        dist_hyp = self.dist_embd(w1embd, w2embd)

        # compare it with the desired distance, which is our loss
        return self.loss(dist_hyp, torch.Tensor(dist_true).to(DEVICE))

    def evaluate_dev_loss(self, data_dev,):
        self.eval()
        losses = []

        for (w1, w1f) in tqdm.tqdm(data_dev):
            # pick w2 at random
            w2, w2f = random.choices(data_dev, k=1)[0]

            dist_true = self.panphon_distance(w1, w2)
            # create micro-batches of one element
            # TODO: create proper batches here
            loss = self.train_step([w1f], [w2f], [dist_true])
            losses.append(loss.cpu().detach())

        return np.average(losses)

    def evaluate_corr(self, data, key=None, batch_size=1024):
        print("Evaluating correlation for", key)

        self.eval()
        data_loader = DataLoader(
            data, batch_size=batch_size, shuffle=False,
            collate_fn=lambda x: [y[1] for y in x]
        )

        # compute embeddings in batches on GPU
        data_embd = [
            self.forward(x).cpu().detach().numpy()
            for x in data_loader
        ]

        # flatten
        data_embd = [x for y in data_embd for x in y]
        data_sims = cosine_distances(data_embd)
        data_sims = np.ravel(data_sims)

        def compute_panphon_distance(y, data):
            fed = panphon2.FeatureTable().feature_edit_distance
            return [fed(x, y) for x, _ in data]

        # scaffolding to compute it only once
        if key is not None and key in self.panphon_dist_cache:
            data_dists_true = self.panphon_dist_cache[key]
        else:
            # parallelization
            with mp.Pool() as pool:
                data_dists_true = pool.map(
                    lambda y: compute_panphon_distance(y[0], data), data)
                # flatten
                data_dists_true = [x for y in data_dists_true for x in y]
        if key is not None and key not in self.panphon_dist_cache:
            self.panphon_dist_cache[key] = data_dists_true

        # compute & return correlations
        parson_corr, _pearson_p = pearsonr(data_sims, data_dists_true)
        spearman_corr, _spearman_p = spearmanr(data_sims, data_dists_true)
        return parson_corr, spearman_corr

    def train_epochs(
        self, data_train, data_dev,
        epochs=1000, batch_size=128,
    ):
        for epoch in range(epochs):
            self.train()

            data_loader = DataLoader(
                data_train, batch_size=batch_size, shuffle=True,
                collate_fn=lambda x: ([y[0] for y in x], [y[1] for y in x])
            )

            losses = []

            for (ws1, ws1f) in tqdm.tqdm(data_loader):
                # pick w2 completely at random
                # TODO: there may be a smarter strategy to do this
                ws2_all = random.choices(data_train, k=len(ws1))
                ws2 = [x[0] for x in ws2_all]
                ws2f = [x[1] for x in ws2_all]

                # compute true distances for the selected pairs
                dist_true = [
                    self.panphon_distance(w1, w2)
                    for w1, w2 in zip(ws1, ws2)
                ]
                loss = self.train_step(ws1f, ws2f, dist_true)

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

            if epoch % 10 == 0:
                corr_dev_pearson, corr_dev_spearman = self.evaluate_corr(
                    data_dev, key="dev"
                )
                # TODO: we don't really need this correlation for train data
                corr_train_pearson, corr_train_spearman = self.evaluate_corr(
                    data_train, key="train"
                )

                print(
                    f"Train pearson  {corr_train_pearson:6.2%}",
                    f"Train spearman {corr_train_spearman:6.2%}",
                )
                print(
                    f"Dev pearson    {corr_dev_pearson:6.2%}",
                    f"Dev spearman   {corr_dev_spearman:6.2%}",
                )
