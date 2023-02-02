import panphon2
import multiprocess as mp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np
import base64

class Evaluator():
    def __init__(self, safe_eval=False):
        self.panphon_dist_cache = {}
        self.panphon_rank_cache = {}
        self.safe_eval = safe_eval

    def evaluate_corr(self, data, data_sims, key=None):
        # TODO: change from microaverage to macroaverage
        # flatten
        data_sims = np.ravel(data_sims)
        # scaffolding to compute it only once
        if key is not None and key in self.panphon_dist_cache:
            data_dists_true = self.panphon_dist_cache[key]
        else:
            def _compute_panphon_distance(y, data):
                fed = panphon2.FeatureTable().feature_edit_distance
                return [fed(tok_ipa, y) for tok_ipa in data]
                
            # parallelization
            with mp.Pool() as pool:
                # tok_features break pipe in multiprocess
                data_ipa = [x[1] for x in data]
                if self.safe_eval:
                    iterator = map
                else:
                    iterator = pool.map
                data_dists_true = list(iterator(
                    lambda y: (
                        _compute_panphon_distance(y, data_ipa)
                    ),
                    data_ipa
                ))
        if key is not None and key not in self.panphon_dist_cache:
            self.panphon_dist_cache[key] = data_dists_true

        # flatten
        data_dists_true = [x for y in data_dists_true for x in y]

        # compute & return correlations
        parson_corr, _pearson_p = pearsonr(data_sims, data_dists_true)
        spearman_corr, _spearman_p = spearmanr(data_sims, data_dists_true)
        return parson_corr, spearman_corr

    @staticmethod
    def _compute_neighbour_rank(row):
        return [x[0] for x in sorted(list(enumerate(row)), key=lambda x: x[1])]

    def evaluate_retrieval(self, data, data_sims, key=None):
        # this is guaranteed to be present so we don't need data
        data_dists_true = self.panphon_dist_cache[key]

        # cache
        if key is not None and key in self.panphon_rank_cache:
            data_rank_true = self.panphon_rank_cache[key]
        else:
            with mp.Pool() as pool:
                data_rank_true = pool.map(
                    self._compute_neighbour_rank, data_dists_true
                )
            # take the second element (not first/zeroth - that's always self)
            data_rank_true = np.array([x[1] for x in data_rank_true])
            data_rank_true = data_rank_true.flatten()

        # store it
        if key is not None and key not in self.panphon_rank_cache:
            self.panphon_rank_cache[key] = data_rank_true

        # get neighbour indicies
        data_dists_hyp = euclidean_distances(np.array(data_sims))

        # paralelize
        with mp.Pool() as pool:
            data_rank_hyp = pool.map(
                self._compute_neighbour_rank, data_dists_hyp
            )

        data_rank_hyp = [
            row.index(nearest_neighbour_index)
            for row, nearest_neighbour_index in zip(data_rank_hyp, data_rank_true)
        ]
        return np.average(data_rank_hyp)
