import panphon2
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import multiprocess as mp
from torch.utils.data import DataLoader

class IntrinsicEvaluator:
    def __init__(self):
        self.dist_cache = {}


    def compute_panphon_distance(self, y, data):
        fed = panphon2.FeatureTable().feature_edit_distance
        return [fed(x, y) for x, _ in data]


    def get_panphon_dists(self, data, key, flatten=True):
        # check if dists for key is already stored, if not compute
        if key is not None and key in self.dist_cache:
            data_dists_true = self.dist_cache[key]
        else:
            # parallelization
            with mp.Pool() as pool:
                data_dists_true = pool.map(
                    lambda y: self.compute_panphon_distance(y[0], data), data)
                
        if key is not None and key not in self.dist_cache:
            self.dist_cache[key] = data_dists_true

        if flatten:
            data_dists_true = [x for y in data_dists_true for x in y]

        return data_dists_true


    def get_vector_dists(self, model, data, flatten=True):
        single_data_loader = DataLoader(dataset=data, 
                                        batch_size=1, 
                                        collate_fn=lambda x: [y[1] for y in x])
        data_embd = [
            model(x).cpu().detach().numpy()
            for x in single_data_loader
        ]

        data_embd = [x for y in data_embd for x in y]
        data_sims = euclidean_distances(data_embd)
        if flatten:
            data_sims = np.ravel(data_sims)

        return data_sims


    def compute_corr(self, model, data, key):
        data_dists_true = self.get_panphon_dists(data, key)
        vector_dists = self.get_vector_dists(model, data)

        pearson_corr, _pearson_p = pearsonr(vector_dists, data_dists_true)
        spearman_corr, _spearman_p = spearmanr(vector_dists, data_dists_true)
        return pearson_corr, spearman_corr
    
    def compute_rank(self, model, data, key):
        # compute distances
        panphon_dists = self.get_panphon_dists(data, key, flatten=False)
        vector_dists = self.get_vector_dists(model, data, flatten=False)

        # get nearest neighbor indices
        panphon_nn_indices = [
            [x[0] for x in sorted(list(enumerate(row)), key=lambda x: x[1])]
            for row in panphon_dists
        ]
        vector_nn_indices = [
            [x[0] for x in sorted(list(enumerate(row)), key=lambda x:x[1])]
            for row in vector_dists
        ]

    def compute_nn_precision(self, model, data, key, k_vals=[1,5,10,25,50]):
        # compute distances
        panphon_dists = self.get_panphon_dists(data, key, flatten=False)
        vector_dists = self.get_vector_dists(model, data, flatten=False)

        # get nearest neighbor indices
        panphon_nn_indices = [
            [x[0] for x in sorted(list(enumerate(row)), key=lambda x: x[1])]
            for row in panphon_dists
        ]
        vector_nn_indices = [
            [x[0] for x in sorted(list(enumerate(row)), key=lambda x:x[1])]
            for row in vector_dists
        ]

        score_dict = {}
        for k in k_vals:
            score_dict[f'acc @ k={k}'] = np.average([
                    x[1] in y[1:k+1] 
                    for x, y in zip(panphon_nn_indices, vector_nn_indices)
                ])

        return score_dict









        

