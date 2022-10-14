import os

import panphon.distance
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import time
import pickle
from tqdm import tqdm


class IntrinsicEvaluator:
    """
      Params:
        phon_feats: list of ipa strings
        phon_embs: array of learned phonological embeddings
    """

    def __init__(self, phon_feats=None, phon_embs=None):
        self.set_phon_feats(phon_feats)
        self.set_phon_embs(phon_embs)

    def set_phon_feats(self, phon_feats):
        self.phon_feats = phon_feats
        self.feat_edit_dist = None

    def set_phon_embs(self, phon_embs):
        self.phon_embs = phon_embs
        self.cos_dist = None

    def feature_edit_distance(self):
        if self.feat_edit_dist is None:
            pickle_path = f"data/feature_edit_dist_{len(self.phon_feats)}.pickle"
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    self.feat_edit_dist = pickle.load(f)
            else:
                t0 = time.time()
                dst = panphon.distance.Distance()
                distances = []
                for i in tqdm(range(len(self.phon_feats))):
                    row = []
                    for j in range(len(self.phon_feats)):
                        row.append(dst.feature_edit_distance(self.phon_feats[i], self.phon_feats[j]))
                    distances.append(row)
                distances = np.ravel(np.array(distances))

                # normalize for fair comparison against cosine distance?
                distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
                self.feat_edit_dist = distances
                t1 = time.time()
                print("feature edit distance took", t1-t0, "seconds")

                # save to pickle
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.feat_edit_dist, f)

    def cosine_distance(self):
        if self.cos_dist is None:
            t0 = time.time()
            distances = cosine_similarity(self.phon_embs)
            distances = np.ravel(distances)
            self.cos_dist = distances
            t1 = time.time()
            print("cosine distance took", t1-t0, "seconds")


    def run(self):
        # compute pairwise feature edit distance of self.phon_feats using the
        # panphon.distance.Distance.feature_edit_distance module
        self.feature_edit_distance()

        # compute pairwise cosine distances between phonological embeddings
        self.cosine_distance()

        # calculate pearson R correlation coefficient
        # print(len(cos_dist), len(feat_edit_dist))
        if self.cos_dist is None or self.feat_edit_dist is None:
            raise ValueError("You must set the embedding and features before running eval")
        pearson_corr, _ = pearsonr(self.cos_dist, self.feat_edit_dist)
        spearman_corr, _ = spearmanr(self.cos_dist, self.feat_edit_dist)

        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }
