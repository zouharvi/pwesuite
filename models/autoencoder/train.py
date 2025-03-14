#!/usr/bin/env python3

import random
import torch
import argparse
from model import RNNAutoencoder
from models.metric_learning.preprocessor import preprocess_dataset

args = argparse.ArgumentParser()
args.add_argument("-l", "--lang", default="en")
args.add_argument(
    "-smp", "--save-model-path",
    default="computed/models/rnn_autoencoder_en.pt"
)
args.add_argument("-e", "--epochs", type=int, default=100)
args.add_argument(
    "-nk", "--number-thousands", type=int, default=200,
    help="Number of training data to use (in thousands) for training",
)
args.add_argument(
    "--eval-train-full", action="store_true",
    help="Compute correlations also for full the training data instead of just 1k sample. This will be significantly slower."
)
args.add_argument("--features", default="panphon")
args.add_argument("--dimension", type=int, default=300)
args = args.parse_args()
random.seed(0)

data = preprocess_dataset(args.features, args.lang)
data_dev = data[:1000]
data = data[1000:]
data_train = random.sample(
    data,
    k=min(args.number_thousands * 1000, len(data))
)
# free up memory
del data

print(f"Loaded {len(data_train)//1000}k words for training")

# target_metric="ip" is not good for some reason
model = RNNAutoencoder(
    dimension=args.dimension,
    feature_size=data_train[0][0].shape[1],
    safe_eval=args.lang == "all" and args.features in {"token_ort", "token_ipa"}
)
model.train_epochs(
    data_train=data_train, data_dev=data_dev,
    eval_train_full=args.eval_train_full,
    epochs=args.epochs
)

if args.save_model_path is not None:
    torch.save(model.state_dict(), args.save_model_path)
