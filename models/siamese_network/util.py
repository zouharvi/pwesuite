import umap
import numpy as np
import torch
import random
from torch.utils.data import DataLoader


def seed_everything(seed: int=sum(bytes(b'dragn'))) -> None:
    """
    Helper function to set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def triplet_collate_fn(batch):
    anchor_feature_array = [b['anchor'][1] for b in batch]
    positive_feature_array = [b['positive'][1] for b in batch]
    stacked_negatives = [b['negative'][1] for b in batch]
    # flatten negatives
    negative_feature_array = [fv for fv_list in stacked_negatives for fv in fv_list]

    return {
        'anchor': anchor_feature_array,
        'positive': positive_feature_array,
        'negative': negative_feature_array
    }


def plain_collate_fn(batch):
    ipa_tokens = [b['ipa'] for b in batch]
    feature_array = [b["feat_vec"] for b in batch]

    return {
        "ipa": ipa_tokens,
        "feat_vec": feature_array
    }


def get_embeddings(model, data):
    # extract embeddings
    single_data_loader = DataLoader(dataset=data, 
                                    batch_size=1, 
                                    collate_fn=lambda x: [y[1] for y in x])
    embeddings = np.array([model(x).cpu().detach().numpy().squeeze() for x in single_data_loader])

    return embeddings



