import argparse
import os

import torch
from torch.utils.data import DataLoader

from dataset import IPATokenDataset
from model.rnn_vae import RNN_VAE
from util import collate_fn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpuid = os.environ.get('CUDA_VISIBLE_DEVICES', 0)
print('device is', device, 'gpu id is', gpuid)


@torch.no_grad()
def inference(loader, vae_model):
    vae_model.eval()
    result = []
    for i, data in enumerate(loader):
        tokens = data['tokens'].to(device)
        feature_array = data['feature_array'].to(device)
        mu, logvar, decoder_logits = vae_model(tokens[:, :-1], feature_array)
        result.append(mu.squeeze(0).cpu().numpy())
    return np.concatenate(result)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=512)
    return parser.parse_args()


if __name__ == '__main__':
    inference_args = parse_args()
    saved_info = torch.load(inference_args.model_path, map_location=device)
    state_dict = saved_info['model']
    args = saved_info['args']
    ipa_vocab = saved_info['ipa_vocab']
    vae_model = RNN_VAE(vocab_size=len(ipa_vocab),
                        emb_dim=24,
                        encoder_hidden_dim=args.encoder_hidden_dim,
                        decoder_hidden_dim=args.decoder_hidden_dim,
                        decoder_input_dim=args.decoder_input_dim,
                        ).to(device)

    inference_dset = IPATokenDataset([inference_args.input_path], ipa_vocab)
    inference_loader = DataLoader(inference_dset, shuffle=False, batch_size=inference_args.batch_size,
                                  collate_fn=collate_fn)
    result = inference(inference_loader, vae_model)
    os.makedirs(os.path.dirname(inference_args.output_path), exist_ok=True)
    np.save(inference_args.output_path, result)
    print("wrote to file:", inference_args.output_path)