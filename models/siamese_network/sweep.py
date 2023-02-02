import argparse
from tqdm import tqdm
import wandb

import panphon2
import numpy as np

import torch
from torch.utils.data import DataLoader

from models import LSTM_Encoder
from dataset import TripletDataset, PlainDataset
from triplet_runner import TripletRunner
from evaluators import IntrinsicEvaluator
from util import triplet_collate_fn, plain_collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
python3 sweep.py --data-file ../data/ipa_tokens_sw.txt \
                 --n-epochs 50 \
                 --n-thousand-train 10 \
                 --n-val 500 \
                 --normalize 0 \
                 --use-proj-head 0 \
                 --wandb-entity natbcar \
                 --wandb-name siamese-network 
"""
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--data-file",
                      type=str,
                      help="path to training file of ipa strings")
    args.add_argument("--n-thousand-train",
                      type=int,
                      default=99,
                      help="number of lines (in thousands) to use during training")
    args.add_argument("--n-val",
                      type=int,
                      default=500,
                      help="number of lines to use during validation")
    args.add_argument("--model-outfile",
                      type=str,
                      default="",
                      help="file to save model to")
    args.add_argument("--embs-outfile",
                      type=str,
                      default="",
                      help="file to save embeddings to")
    args.add_argument("--wandb-name",
                      type=str,
                      default="",
                      help="name of wandb run")
    args.add_argument("--wandb-entity", 
                      type=str, 
                      default="natbcar")
    args.add_argument("--n-epochs",
                      type=int,
                      default=50,
                      help="number of epochs to train for")
    args.add_argument("--train-batch-size",
                      type=int,
                      default=64,
                      help="training batch size")
    args.add_argument("--val-batch-size",
                      type=int,
                      default=128,
                      help="validation batch size")
    args.add_argument("--hidden-size",
                      type=int,
                      default=64,
                      help="model hidden dimension")
    args.add_argument("--num_layers",
                      type=int,
                      default=2,
                      help="number of layers in LSTM encoder")
    args.add_argument("--dropout",
                      type=int,
                      default=0.3,
                      help="dropout probability for LSTM encoder")
    args.add_argument("--lr",
                      type=float,
                      default=1e-4,
                      help="learning rate")
    args.add_argument("--margin",
                      type=float,
                      default=1.0,
                      help="margin for triplet loss calculation")
    args.add_argument("--use-attn",
                      type=int,
                      default=1,
                      help="margin for triplet loss calculation")
    args.add_argument("--use-proj-head",
                      type=int,
                      default=1,
                      help="margin for triplet loss calculation")
    args.add_argument("--normalize",
                      type=int,
                      default=0,
                      help="to normalize the encoder outputs before computing loss")
    args.add_argument("--ord",
                      type=int,
                      default=2,
                      help="norm for pairwise distance in triplet loss")
    args.add_argument("--checkpoint-file",
                      type=str,
                      default="",
                      help="resume model checkpoint file")
    return args.parse_args()
    
def load_data(args):
    with open(args.data_file, "r") as f:
        data = [x.rstrip("\n") for x in f.readlines()][:args.n_val+args.n_thousand_train*1000]
    
    print(f"Loaded {len(data)//1000}k words")

    ft = panphon2.FeatureTable()
    data = [(w, ft.word_to_binary_vectors(w)) for w in tqdm(data)]
    np.random.shuffle(data)

    data_val = data[:args.n_val]
    data_train = data[args.n_val:]

    return data_train, data_val

def main():
    args = parse_args()

    # name = f'attn={wandb.config.use_attn} lr={wandb.config.lr} margin={wandb.config.margin} hidden_size={wandb.config.hidden_size} batch_size={wandb.config.batch_size}'
    run = wandb.init(project='test-sweep', entity=args.wandb_entity)

    # encoder & loss fn & optimizer
    encoder = LSTM_Encoder(hidden_size=wandb.config.hidden_size, 
                            num_layers=args.num_layers, 
                            dropout=args.dropout, 
                            device=DEVICE,
                            normalize=args.normalize,
                            use_attn=wandb.config.use_attn,
                            use_proj_head=args.use_proj_head)
    criterion = torch.nn.TripletMarginLoss(margin=wandb.config.margin, p=args.ord)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=wandb.config.lr)

    # data
    data_train, data_val = load_data(args)
    train_dataset = PlainDataset(data=data_train)
    val_dataset = PlainDataset(data=data_val)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=wandb.config.batch_size,
                              shuffle=False, 
                              collate_fn=plain_collate_fn) 
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=wandb.config.batch_size,
                            shuffle=False, 
                            collate_fn=plain_collate_fn)

    # evaluator
    evaluator = IntrinsicEvaluator()

    runner = TripletRunner(model=encoder,
                        criterion=criterion,
                        optimizer=optimizer,
                        data_train=data_train,
                        data_val=data_val,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        evaluator=evaluator,
                        n_epochs=args.n_epochs,
                        wandb_name=args.wandb_name,
                        wandb_entity=args.wandb_entity)
    runner()

if __name__ == "__main__":
    # sweep_configuration = {
    #     'method': 'grid',
    #     'name': 'sweep',
    #     'metric': {'goal': 'maximize', 'name': 'acc @ k=1'},
    #     'parameters': 
    #     {
    #         'use_attn': {'values': [True, False]},
    #         'lr': {'values': [1e-2, 1e-3, 1e-4]},
    #         'margin': {'values': [0.1, 0.5, 1.0, 1.5]},
    #         'hidden_size': {'values': [64, 128, 256]},
    #         'batch_size': {'values': [128, 256]}
    #     }
    # }

    sweep_configuration = {
        'method': 'grid',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'acc @ k=1'},
        'parameters': 
        {
            'use_attn': {'values': [True]},
            'lr': {'values': [1e-2, 1e-3]},
            'margin': {'values': [0.1, 0.5]},
            'hidden_size': {'values': [128, 256]},
            'batch_size': {'values': [128, 256]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project='test-sweep')
    wandb.agent(sweep_id, function=main)