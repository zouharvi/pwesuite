# Imports are done within functions so that they are not needlessly loaded when a different function is used

def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def collate_fn(batch):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from vocab import PAD_IDX

    feature_array = [torch.tensor(b['feature_array']) for b in batch]
    tokens = [torch.tensor(b['tokens']) for b in batch]
    feature_array = pad_sequence(
        feature_array, padding_value=PAD_IDX, batch_first=True)
    tokens = pad_sequence(tokens, padding_value=PAD_IDX, batch_first=True)

    return {
        'feature_array': feature_array.float(),
        'tokens': tokens,
    }


def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = std.data.new(std.size()).normal_()
    return eps.mul(std).add_(mu)


def get_kl_loss(mu, logvar):
    import torch
    if logvar is None:
        return torch.tensor(0., device=mu.device)
    kl_loss = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    return kl_loss


def save_model(model, optimizer, args, ipa_vocab, epoch, filepath):
    import torch
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'epoch': epoch,
        'ipa_vocab': ipa_vocab,
    }
    torch.save(save_info, filepath)
    print(f'\t>> saved model to {filepath}')
