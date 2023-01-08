import os
import time

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import wandb
from dataset import IPATokenDataset
from intrinsic_eval import IntrinsicEvaluator
from vocab import *
from model.language_model import AutoregressiveLM
from util import *
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
print('gpu id is', gpuid)
model_save_path = f'./checkpoints/gpu{gpuid}_best_loss.pt'


def train_step(model, train_loader, optimizer, objective, limit_iter_per_epoch=None):
    model.train()

    total_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        tokens = batch['tokens'].to(device)                 # (B, S)
        feature_matrix = batch['feature_array'].to(device)  # (B, S, 24)
        batch_size = tokens.size(0)
        num_segments = feature_matrix.size(1)  # EXcludes <bos> and <eos>

        logits = []
        # the <BOS> embedding is an embedding layer with a vocab of 1 (the <BOS> token)
        #   (B, 1, 24)
        prev_token_feats = model.bos_embedding(torch.zeros(batch_size, 1, dtype=torch.int64).to(device))
        # initialize hidden and cell state to 0s
        (hidden, cell) = \
            (torch.zeros((2 if model.bidirectional else 1) * model.num_layers, batch_size, model.hidden_dim).to(device),
            torch.zeros((2 if model.bidirectional else 1) * model.num_layers, batch_size, model.hidden_dim).to(device))

        # + 1 because we feed in <BOS> first
        for i in range(num_segments + 1):
            # LM objective: feed in previous token, predict current token
            logit, (hidden, cell) = model(prev_token_feats, hidden, cell)
            logits.append(logit)

            # teacher forcing - the correct previous token is fed in as input
            # i starts at 0 and corresponds to the first segment (not <BOS>)
            # note that feature_matrix excludes <BOS> and <EOS>
            if i < num_segments:
                prev_token_feats = feature_matrix[:, i, :].unsqueeze(dim=1)

        logits = torch.cat(logits, dim=1)
        logits = logits.transpose(1, 2)  # (B, V, S) as expected by CrossEntropyLoss
        # the input sequence left shifted over by 1, which removes <BOS> but keeps <EOS>
        target = tokens[:, 1:]
        loss = objective(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if limit_iter_per_epoch is not None and i >= limit_iter_per_epoch:
            # this is nothing more than a way to log more frequently than every (full) epoch.
            break

    N = len(train_loader)
    return {
        'train/loss': total_loss / N
    }


@torch.no_grad()
def validate_step(model, val_loader, objective, evaluator):
    model.eval()
    pooled_phon_embs = []  # for intrinsic evaluation

    total_loss = 0
    for batch in val_loader:
        tokens = batch['tokens'].to(device)
        feature_matrix = batch['feature_array'].to(device)
        batch_size = tokens.size(0)
        num_segments = feature_matrix.size(1)  # EXcludes <bos> and <eos>

        # intrinsic evaluation - first obtain a pooled embedding (the useful part)
        bos_embedding = model.bos_embedding(torch.zeros(batch_size, 1, dtype=torch.int64).to(device))  # (B, 1, 24)
        feature_matrix = torch.cat((bos_embedding, feature_matrix), dim=1)  # (B, S + 1, 24)
        # take the LSTM output corresponding to the final token
        # TODO: concatenate with LSTM output corresponding to <BOS> cuz it's bidirectional??
        embeddings = model.pool(feature_matrix)
        pooled_phon_embs.append(embeddings)

        # language modeling objective - see train_step() for more details
        logits = []
        prev_token_feats = model.bos_embedding(torch.zeros(batch_size, 1, dtype=torch.int64).to(device))
        (hidden, cell) = \
            (torch.zeros((2 if model.bidirectional else 1) * model.num_layers, batch_size, model.hidden_dim).to(device),
             torch.zeros((2 if model.bidirectional else 1) * model.num_layers, batch_size, model.hidden_dim).to(device))
        for i in range(num_segments + 1):
            logit, (hidden, cell) = model(prev_token_feats, hidden, cell)
            logits.append(logit)
            if i < num_segments:
                prev_token_feats = feature_matrix[:, i, :].unsqueeze(dim=1)
        logits = torch.cat(logits, dim=1)
        logits = logits.transpose(1, 2)
        target = tokens[:, 1:]
        loss = objective(logits, target)
        total_loss += loss.item()

    # during training, we calculate the MSE between feature edit distance and the cosine similarity of 2 vectors
    # during evaluation, Pearson's/Spearman's correlation coefficient is used instead
    # evaluator assumes that phon_feats and phon_embs are in the same order
    evaluator.set_phon_embs(torch.cat(pooled_phon_embs, dim=0).detach().cpu().numpy())
    intrinsic_eval = evaluator.run()

    N = len(val_loader)
    return {
        'val/loss': total_loss / N,
        'val/intrinsic_pearson_correlation': intrinsic_eval['pearson'],
        'val/intrinsic_spearman_correlation': intrinsic_eval['spearman'],
    }


def train(args, vocab, model):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, eps=1e-9)
    objective = CrossEntropyLoss(ignore_index=PAD_IDX)

    # TODO: sort the sequences by length? to avoid really bad padding scenarios

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True, 'collate_fn': collate_fn}
    train_dset = IPATokenDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab,
                                 split_bounds=(0, args.train_ratio))
    train_loader = DataLoader(train_dset, shuffle=True, **loader_kwargs)
    val_dset = IPATokenDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab,
                               split_bounds=(args.train_ratio, 1.0))
    val_loader = DataLoader(val_dset, shuffle=False, **loader_kwargs)
    best_intrinsic = 0
    evaluator = IntrinsicEvaluator()
    # list of IPA transcriptions for each word in the val dataset
    evaluator.set_phon_feats([d['ipa'] for d in val_dset])

    # find out the correlation before any training
    val_loss_dict = validate_step(model, val_loader, objective, evaluator)
    spearman = val_loss_dict["val/intrinsic_spearman_correlation"]
    pearson = val_loss_dict["val/intrinsic_pearson_correlation"]
    wandb.log({"val/initial_spearman_correlation": spearman})
    wandb.log({"val/initial_pearson_correlation": pearson})
    print("Initial spearman correlation is,", spearman)
    print("Initial pearson correlation is,", pearson)

    for ep in range(args.epochs):
        t = time.time()

        train_loss_dict = train_step(model, train_loader, optimizer, objective,
                                     limit_iter_per_epoch=args.limit_iter_per_epoch)
        train_time = time.time()
        val_loss_dict = validate_step(model, val_loader, objective, evaluator)
        best_intrinsic = max(best_intrinsic, val_loss_dict["val/intrinsic_spearman_correlation"])
        val_loss_dict["val/BEST_intrinsic_spearman_correlation"] = best_intrinsic
        wandb.log({"train/lr": optimizer.param_groups[0]['lr'], **train_loss_dict, **val_loss_dict})

        spearman = val_loss_dict["val/intrinsic_spearman_correlation"]
        if abs(spearman) > abs(best_intrinsic):
            best_intrinsic = spearman
            save_model(model, optimizer, args, ipa_vocab, ep, model_save_path)

        print(f'< epoch {ep} >  (elapsed: {time.time() - t:.2f}s, decode time: {time.time() - train_time:.2f}s)')
        print(f'  * [train]  loss: {train_loss_dict["train/loss"]:.6f}')
        print(f'  * [ val ]  loss: {val_loss_dict["val/loss"]:.6f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_codes', nargs='+')
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bidirectional', type=str2bool, default=True)
    parser.add_argument('--num_layers', help='number of LSTM layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--limit_iter_per_epoch', type=int, default=200)
    parser.add_argument('--wandb_name', type=str, default="")
    parser.add_argument('--wandb_entity', type=str, default="kalvin")
    parser.add_argument('--sweeping', type=str2bool, default=False)
    parser.add_argument('--train_ratio', type=float, default=0.999)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="phonological-pooling", name=args.wandb_name, entity=args.wandb_entity,
               mode='disabled' if (not args.wandb_name and not args.sweeping) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py'))
    wandb.config.update(args)
    os.makedirs("checkpoints", exist_ok=True)

    # TODO: find a better way to get the dims from panphon
    PANPHON_FEATURE_DIM = 24

    ipa_vocab = Vocab(tokens_file=args.vocab_file)
    model = AutoregressiveLM(
        feature_size=PANPHON_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        vocab_size=len(ipa_vocab),
    ).to(device)
    train(args, ipa_vocab, model)
