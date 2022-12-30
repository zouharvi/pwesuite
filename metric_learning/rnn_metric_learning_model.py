import panphon2
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_distances
import random
import tqdm
import numpy as np
from intrinsic_evaluator import Evaluator

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
        self.batch_size_eval = 2048
        self.batch_size_train = 128

        # TODO: try to use characters instead of vectors but maybe that doesn't matter
        self.panphon_vectors = panphon_vectors

        # TODO: contrastive learning
        self.loss = torch.nn.MSELoss()
        self.panphon_distance = panphon2.FeatureTable().feature_edit_distance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

        if target_metric == "l2":
            self.dist_embd = torch.nn.PairwiseDistance(p=2)
        elif target_metric == "cos":
            self.dist_embd = torch.nn.CosineSimilarity()
        else:
            raise Exception(f"Unknown metric {target_metric}")

        self.evaluator = Evaluator()

        # move the model to GPU
        self.to(DEVICE)

    def evaluate_intrinsic(self, data, key=None):
        print("Evaluating everything for", key)

        self.eval()
        data_loader = DataLoader(
            data, batch_size=self.batch_size_eval, shuffle=False,
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
        # data_sims = np.ravel(data_sims)

        pearson_corr, spearman_corr = self.evaluator.evaluate_corr(
            data, data_sims, key=key
        )

        retrieval_rank = self.evaluator.evaluate_retrieval(
            data, data_sims, key=key
        )
        out_str = f"ρp={pearson_corr:.0%} ρs={spearman_corr:.0%} rank={retrieval_rank:.0f}"
        return out_str

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

    def evaluate_dev_loss(self, data_dev):
        self.eval()
        losses = []

        data_loader = DataLoader(
            data_dev, batch_size=self.batch_size_eval, shuffle=True,
            collate_fn=lambda x: ([y[0] for y in x], [y[1] for y in x])
        )

        for (ws1, ws1f) in tqdm.tqdm(data_loader):
            # pick w2 at random
            ws2_all = random.choices(data_dev, k=len(ws1))
            ws2 = [x[0] for x in ws2_all]
            ws2f = [x[1] for x in ws2_all]

            dist_true = [
                self.panphon_distance(w1, w2)
                for w1, w2 in zip(ws1, ws2)
            ]
            # create micro-batches of one element
            # TODO: create proper batches here
            loss = self.train_step(ws1f, ws2f, dist_true)
            losses.append(loss.cpu().detach())

        return np.average(losses)

    def train_epochs(
        self, data_train, data_dev,
        epochs=1000, eval_train_full=False,
    ):
        for epoch in range(epochs):
            self.train()

            data_loader = DataLoader(
                data_train, batch_size=self.batch_size_train, shuffle=True,
                collate_fn=lambda x: ([y[0] for y in x], [y[1] for y in x])
            )

            losses = []

            for (ws1, ws1f) in tqdm.tqdm(data_loader):
                # pick w2 completely at random
                # TODO: there may be a smarter strategy to do this such as prefering local space
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
                eval_dev = self.evaluate_intrinsic(
                    data_dev, key="dev"
                )
                # use only part of the training data for evaluation unless specified otherwise
                eval_train = self.evaluate_intrinsic(
                    data_train if eval_train_full else data_train[:1000], key="train"
                )

                print("Train", eval_train, "|||", "dev", eval_dev)