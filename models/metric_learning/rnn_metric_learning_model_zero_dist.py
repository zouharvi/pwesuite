import numpy as np
import panphon2
import multiprocess as mp
import torch
from sklearn.metrics.pairwise import cosine_distances
import tqdm
from rnn_metric_learning_model import RNNMetricLearner
from main.utils import get_device

DEVICE = get_device()

class RNNMetricLearnerZeroDist(RNNMetricLearner):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.beta = 5


    def train_step(self, ws1i, ws1, ws2, dist_true):
        if ws1i is None:
            return super().train_step(ws1i, ws1, ws2, dist_true)

        w1embd = self.forward(ws1)
        w2embd = self.forward(ws2)

        w1neighbours_i = self.data_train_nearest_neighbour_index[ws1i]
        w1neighbours = [self.data_train[x][1] for x in w1neighbours_i]
        w1neighour_embd = self.forward(w1neighbours)

        # compute distance in embd space
        dist_hyp = self.dist_embd(w1embd, w2embd)
        dist_neighbour = self.dist_embd(w1embd, w1neighour_embd)

        # loss_primary = self.loss(dist_hyp, torch.ones_like(dist_neighbour).to(DEVICE))
        loss_primary = self.loss(dist_hyp, torch.Tensor(dist_true).to(DEVICE))
        # force the neighbour to have zero distance
        loss_neighbour = self.loss(dist_neighbour, torch.zeros_like(dist_neighbour).to(DEVICE))

        # compare it with the desired distance, which is our loss
        return loss_primary + self.beta * loss_neighbour


    def train_epochs(
        self, data_train, **kwargs
    ):
        print("Preprocessing training data neighbours")
        def _compute_panphon_distance(y, data):
            fed = panphon2.FeatureTable().feature_edit_distance
            return [fed(x, y) for x, _ in data]
        def _compute_neighbour_rank(row):
            return [x[0] for x in sorted(list(enumerate(row)), key=lambda x: x[1])]

        # parallelization
        with mp.Pool() as pool:
            data_dists_true = pool.map(
                lambda y: _compute_panphon_distance(y[0], data_train), tqdm.tqdm(data_train)
            )
            data_rank_true = pool.map(
                _compute_neighbour_rank, tqdm.tqdm(data_dists_true)
            )
            data_rank_true = np.array([x[1] for x in data_rank_true])

        # ugly global states
        self.data_train = data_train
        self.data_train_nearest_neighbour_index  = data_rank_true.flatten()
        
        super().train_epochs(data_train, **kwargs)
