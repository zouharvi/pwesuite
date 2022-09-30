import os
import time

import transformers
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import wandb
from dataset import IPATokenDataset
from vocab import *
from model import RNN_VAE
from util import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
print('gpu id is', gpuid)
model_save_path = f'./checkpoints/gpu{gpuid}_best_loss.pt'


def train_step(vae_model, train_loader, optimizer, loss_multipliers):
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

    N = len(train_loader)
    return {
        'train/recon_loss': total_recon / N,
        'train/kl_loss': total_kl / N,
        'train/loss': total_loss / N
    }

@torch.no_grad()
def validate_step(vae_model, val_loader, loss_multipliers):
    vae_model.eval()
    cross_entropy = CrossEntropyLoss(ignore_index=PAD_IDX)

    total_kl, total_recon, total_loss = 0, 0, 0
    for i, data in enumerate(val_loader):
        tokens = data['tokens'].to(device)
        feature_array = data['feature_array'].to(device)
        mu, logvar, decoder_logits = vae_model(tokens[:, :-1], feature_array)
        kl_loss = get_kl_loss(mu, logvar)
        recon_loss = cross_entropy(decoder_logits.reshape((-1, decoder_logits.shape[-1])),
                                   tokens[:, 1:].reshape(-1))

        loss = kl_loss * loss_multipliers['kl'] + recon_loss * loss_multipliers['recon']

        total_kl += kl_loss.item()
        total_recon += recon_loss.item()
        total_loss += loss.item()

    N = len(val_loader)
    return {
        'val/recon_loss': total_recon / N,
        'val/kl_loss': total_kl / N,
        'val/loss': total_loss / N
    }


def train(args, vocab, vae_model, loss_multipliers):
    optimizer = torch.optim.Adam(vae_model.parameters(),
                                 lr=args.lr, eps=1e-9)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epochs,
                                                             num_training_steps=args.epochs)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': True, 'collate_fn': collate_fn}
    train_loader = DataLoader(IPATokenDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab,
                                              split_bounds=(0, 0.9)),
                              shuffle=True, **loader_kwargs)
    val_loader = DataLoader(IPATokenDataset([f'data/ipa_tokens_{lang}.txt' for lang in args.lang_codes], vocab,
                                            split_bounds=(0.9, 1.0)),
                            shuffle=False, **loader_kwargs)
    best_val_loss = 1e10

    for ep in range(args.epochs):
        t = time.time()

        train_loss_dict = train_step(vae_model, train_loader, optimizer, loss_multipliers)
        train_time = time.time()
        val_loss_dict = validate_step(vae_model, val_loader, loss_multipliers)
        wandb.log({"train/lr": optimizer.param_groups[0]['lr'], **train_loss_dict, **val_loss_dict})

        val_loss = val_loss_dict["val/loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(vae_model, optimizer, args, ipa_vocab, ep, model_save_path)

        print(f'< epoch {ep} >  (elapsed: {time.time() - t:.2f}s, decode time: {time.time() - train_time:.2f}s)')
        print(f'  * [train]  loss: {train_loss_dict["train/loss"]:.6f}')
        print(f'  * [ val ]  loss: {val_loss_dict["dev/loss"]:.6f}')

        scheduler.step()

def inference(args, vae_model):
    ...


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_codes', nargs='+')
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--encoder_hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_hidden_dim', type=int, default=128)
    parser.add_argument('--decoder_input_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--kl_mult', type=float, default=0.1)
    parser.add_argument('--wandb_name', type=str, default="")
    parser.add_argument('--wandb_entity', type=str, default="cuichenx")
    parser.add_argument('--sweeping', type=str2bool, default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    wandb.init(project="phonetic_repr", name=args.wandb_name, entity=args.wandb_entity,
               mode='disabled' if (not args.wandb_name and not args.sweeping) else 'online')
    wandb.run.log_code(".", include_fn=lambda path: path.endswith('.py'))
    wandb.config.update(args)
    os.makedirs("checkpoints", exist_ok=True)

    ipa_vocab = Vocab(tokens_file='data/vocab.txt')
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

    inference(args, vae_model)
