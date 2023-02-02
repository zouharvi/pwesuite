import argparse
from tqdm import tqdm
import wandb

import panphon2
import numpy as np

import torch
from torch.utils.data import DataLoader

from models import LSTM_Encoder
from dataset import PlainDataset
from triplet_runner import TripletRunner
from evaluators import IntrinsicEvaluator
from util import triplet_collate_fn, plain_collate_fn, seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
python3 train.py --data-file ../data/ipa_tokens_sw.txt \
                 --n-epochs 1 \
                 --embs-outfile test.out \
                 --train-batch-size 256 \
                 --val-batch-size 256 \
                 --n-thousand-train 5 \
                 --n-val 500 \
                 --margin 0.1 \
                 --hidden-size 128 \
                 --lr 1e-3 \
                 --use-attn 1 \
                 --num_layers 2 \
                 --wandb-entity natbcar \
                 --wandb-name test_run
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
                      default=10,
                      help="number of epochs to train for")
    args.add_argument("--eval-every",
                      type=int,
                      default=5,
                      help="evaluate model every X epochs")
    args.add_argument("--train-batch-size",
                      type=int,
                      default=256,
                      help="training batch size")
    args.add_argument("--val-batch-size",
                      type=int,
                      default=256,
                      help="validation batch size")
    args.add_argument("--hidden-size",
                      type=int,
                      default=256,
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
                      default=1e-3,
                      help="learning rate")
    args.add_argument("--margin",
                      type=float,
                      default=0.5,
                      help="margin for triplet loss calculation")
    args.add_argument("--use-attn",
                      type=int,
                      default=1,
                      help="margin for triplet loss calculation")
    args.add_argument("--checkpoint-file",
                      type=str,
                      default="",
                      help="resume model checkpoint file")
    return args.parse_args()
    

def load_data(args):
    seed_everything(42)

    with open(args.data_file, "r") as f:
        data = [x.rstrip("\n") for x in f.readlines()][:args.n_val+args.n_thousand_train*1000]
    
    print(f"Loaded {len(data)//1000}k words")

    ft = panphon2.FeatureTable()
    data = [(w, ft.word_to_binary_vectors(w)) for w in tqdm(data)]
    np.random.shuffle(data)

    data_val = data[:args.n_val]
    data_train = data[args.n_val:]

    return data_train, data_val


def inference(model, data, outfile):
    model.eval()

    f = open(outfile, "w")
    ft = panphon2.FeatureTable()
    count = 0
    for w, x in data:
        x = model([x]).squeeze(0).cpu().detach().numpy()
        emb_str = " ".join([w] + [str(xi) for xi in x])
        f.write(emb_str + "\n")
        count += 1
    f.close()
    
    print(f"{100* (count/len(data)):.2f} % of words embedded")
    

def main():
    args = parse_args()

    # weights and biases
    wandb.init(project="siamese_network", name=args.wandb_name, entity=args.wandb_entity,
            mode='disabled' if (not args.wandb_name) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py'))
    wandb.config.update(args)

    encoder = LSTM_Encoder(hidden_size=args.hidden_size, 
                            num_layers=args.num_layers, 
                            dropout=args.dropout, 
                            device=DEVICE,
                            use_attn=args.use_attn)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    
    # data
    data_train, data_val = load_data(args)
    train_dataset = PlainDataset(data=data_train)
    val_dataset = PlainDataset(data=data_val)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=args.train_batch_size, 
                              shuffle=False, 
                              collate_fn=plain_collate_fn) 
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            shuffle=False, 
                            collate_fn=plain_collate_fn)

    # evaluator
    # TODO: add rank for retrieval 
    evaluator = IntrinsicEvaluator()

    runner = TripletRunner(model=encoder,
                        optimizer=optimizer,
                        margin=args.margin,
                        data_train=data_train,
                        data_val=data_val,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        evaluator=evaluator,
                        n_epochs=args.n_epochs,
                        eval_every=args.eval_every,
                        wandb_name=args.wandb_name,
                        wandb_entity=args.wandb_entity)
    runner()

    # save embeddings
    data = data_train + data_val
    inference(runner.model, data, args.embs_outfile)

if __name__ == "__main__":
    main()