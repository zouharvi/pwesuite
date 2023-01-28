#!/usr/bin/env python3
 
import epitran
import panphon2
import torch
from models.metric_learning.model import RNNMetricLearner

# process data
f = panphon2.FeatureTable()
epi = epitran.Epitran("eng-Latn")

model = RNNMetricLearner(target_metric="l2")
model.load_state_dict(torch.load("computed/models/model_pl.pt"))

for word in ["write", "cite"]:
    print(word)
    word = epi.transliterate(word)
    print(word)
    word = f.word_to_binary_vectors(word)
    print(word)
    word = model.forward([word]).detach().cpu().numpy()
    print(word)
    print()
