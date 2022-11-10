import os
import time

import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import wandb
from dataset import IPATokenDataset
from intrinsic_eval import IntrinsicEvaluator
from vocab import *
from model.rnn_vae import RNN_VAE
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
print('gpu id is', gpuid)
model_save_path = f'./checkpoints/gpu{gpuid}_best_loss.pt'


def train_step(vae_model, train_loader, optimizer, loss_multipliers, limit_iter_per_epoch=None):
    vae_model.train()
    cross_entropy = CrossEntropyLoss(ignore_index=PAD_IDX)

    total_kl, total_recon, total_loss = 0, 0, 0
    for i, data in enumerate(train_loader):
        tokens = data['tokens'].to(device)
        feature_array = data['feature_array'].to(device)
        mu, logvar, decoder_logits = vae_model(tokens[:, :-1], feature_array)
        kl_loss = get_kl_loss(mu, logvar)
        recon_loss = cross_entropy(decoder_logits.reshape((-1, decoder_logits.shape[-1])),
                         tokens[:, 1:].reshape(-1))

        loss = kl_loss * loss_multipliers['kl'] + recon_loss * loss_multipliers['recon']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_kl += kl_loss.item()
        total_recon += recon_loss.item()
        total_loss += loss.item()
        if limit_iter_per_epoch is not None and i >= limit_iter_per_epoch:
            # this is nothing more than a way to log more frequently than every (full) epoch.
            break

    N = i+1  # len(train_loader)
    return {
        'train/recon_loss': total_recon / N,
        'train/kl_loss': total_kl / N,
        'train/loss': total_loss / N
    }

@torch.no_grad()
def validate_step(vae_model, val_loader, loss_multipliers, evaluator):
    vae_model.eval()
    cross_entropy = CrossEntropyLoss(ignore_index=PAD_IDX)
    phon_embs = []
    total_kl, total_recon, total_loss = 0, 0, 0
    for i, data in enumerate(val_loader):
        tokens = data['tokens'].to(device)
        feature_array = data['feature_array'].to(device)
        mu, logvar, decoder_logits = vae_model(tokens[:, :-1], feature_array)
        phon_embs.append(mu[0])
        kl_loss = get_kl_loss(mu, logvar)
        recon_loss = cross_entropy(decoder_logits.reshape((-1, decoder_logits.shape[-1])),
                                   tokens[:, 1:].reshape(-1))

        loss = kl_loss * loss_multipliers['kl'] + recon_loss * loss_multipliers['recon']

        total_kl += kl_loss.item()
        total_recon += recon_loss.item()
        total_loss += loss.item()


    evaluator.set_phon_embs(torch.cat(phon_embs, 0).detach().cpu().numpy())
    intrinsic_eval = evaluator.run()

    N = len(val_loader)
    return {
        'val/recon_loss': total_recon / N,
        'val/kl_loss': total_kl / N,
        'val/loss': total_loss / N,
        'val/intrinsic_pearson_correlation': intrinsic_eval['pearson'],
        'val/intrinsic_spearman_correlation': intrinsic_eval['spearman'],
    }


def train(args, vocab, vae_model, loss_multipliers):
    optimizer = torch.optim.Adam(vae_model.parameters(),
                                 lr=args.lr, eps=1e-9)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epochs,
                                                             num_training_steps=args.epochs)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True, 'collate_fn': collate_fn}
    train_dset = IPATokenDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab, split_bounds=(0, args.train_ratio))
    train_loader = DataLoader(train_dset, shuffle=True, **loader_kwargs)
    val_dset = IPATokenDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab, split_bounds=(args.train_ratio, 1.0))
    val_loader = DataLoader(val_dset, shuffle=False, **loader_kwargs)
    best_val_loss = 1e10
    evaluator = IntrinsicEvaluator()
    evaluator.set_phon_feats([d['ipa'] for d in val_dset])

    for ep in range(args.epochs):
        t = time.time()

        train_loss_dict = train_step(vae_model, train_loader, optimizer, loss_multipliers,
                                     limit_iter_per_epoch=args.limit_iter_per_epoch)
        train_time = time.time()
        val_loss_dict = validate_step(vae_model, val_loader, loss_multipliers, evaluator)

        wandb.log({"train/lr": optimizer.param_groups[0]['lr'], **train_loss_dict, **val_loss_dict})

        val_loss = val_loss_dict["val/loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(vae_model, optimizer, args, ipa_vocab, ep, model_save_path)

        print(f'< epoch {ep} >  (elapsed: {time.time() - t:.2f}s, decode time: {time.time() - train_time:.2f}s)')
        print(f'  * [train]  loss: {train_loss_dict["train/loss"]:.6f}')
        print(f'  * [ val ]  loss: {val_loss_dict["val/loss"]:.6f}')

        scheduler.step()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_codes', nargs='+')
    parser.add_argument('--vocab_file', type=str)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--encoder_hidden_dim', type=int, default=16)
    parser.add_argument('--decoder_hidden_dim', type=int, default=16)
    parser.add_argument('--decoder_input_dim', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--kl_mult', type=float, default=0.3)
    parser.add_argument('--limit_iter_per_epoch', type=int, default=200)
    parser.add_argument('--wandb_name', type=str, default="")
    parser.add_argument('--wandb_entity', type=str, default="cuichenx")
    parser.add_argument('--sweeping', type=str2bool, default=False)
    parser.add_argument('--train_ratio', type=float, default=0.999)
    return parser.parse_args()

if __name__ == '__main__':
    torch.set_num_threads(4)
    args = parse_args()
    wandb.init(project="phonetic_repr", name=args.wandb_name, entity=args.wandb_entity,
               mode='disabled' if (not args.wandb_name and not args.sweeping) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py'))
    wandb.config.update(args)
    os.makedirs("checkpoints", exist_ok=True)

    ipa_vocab = Vocab(tokens_file=args.vocab_file)
    vae_model = RNN_VAE(vocab_size=len(ipa_vocab),
                        emb_dim=24,
                        encoder_hidden_dim=args.encoder_hidden_dim,
                        decoder_hidden_dim=args.decoder_hidden_dim,
                        decoder_input_dim=args.decoder_input_dim,
                        ).to(device)
    loss_multipliers = {
        'recon': 1,
        'kl': args.kl_mult
    }

    train(args, ipa_vocab, vae_model, loss_multipliers)

