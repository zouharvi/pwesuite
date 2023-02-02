from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import multiprocess as mp
import random
import gc
import panphon2
import wandb
from util import seed_everything

class TripletRunner:
    def __init__(self, model, optimizer, margin, data_train, data_val, train_loader, val_loader, evaluator, n_epochs, eval_every, wandb_name="", wandb_entity=""):
        seed_everything(42)

        self.model = model

        # loss stuff and optimizer
        self.margin = margin
        self.optimizer = optimizer
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=50, verbose=True)

        # data
        self.data_train = data_train
        self.data_val = data_val
        self.train_loader = train_loader
        self.val_loader = val_loader

        # evaluator
        self.evaluator = evaluator
        self.n_epochs = n_epochs
        self.eval_every = eval_every
        
        # to cache triplet indices and panphon distances
        self.panphon_distance_cache = {}
        self.batch_triplet_cache = {}

        # dist fns to compute panphon and vector distances
        self.fed = panphon2.FeatureTable().feature_edit_distance
        self.dists_emb = torch.nn.PairwiseDistance(p=2)


    def clear_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


    def compute_batch_panphon_distance(self, data):
        def compute_panphon_dist(y, data):
            fed = panphon2.FeatureTable().feature_edit_distance
            return [fed(x, y) for x in data]

        with mp.Pool() as pool:
            distances = pool.map(lambda y: compute_panphon_dist(y, data), data)
        distances = np.array(distances)
        return distances


    def get_batch_triplets(self, data, key, batch_size):
        if key not in self.batch_triplet_cache:
            # create mask filter only distinct triplets [i,j,k]
            mask = np.ones((batch_size, batch_size))
            np.fill_diagonal(mask, 0)
            i_neq_j = np.expand_dims(mask, 2)
            i_neq_k = np.expand_dims(mask, 1)
            j_neq_k = np.expand_dims(mask, 0)
            distinct_mask = np.logical_and(np.logical_and(i_neq_j, i_neq_k), j_neq_k)

            # matrix of panphon distances between ipa strings
            # CHANGED: we only do this on the first epoch then cache the valid triplet indices
            # so there is no need to save the distances and it wastes space
            panphon_distances = self.compute_batch_panphon_distance(data=data)

            # # get matrix of all possible distances d(a,p) - d(a,n) + margin
            distance_diff_mat = np.expand_dims(panphon_distances, 2) - np.expand_dims(panphon_distances, 1)
            # mask distance matrix 
            valid_triplets = distinct_mask * distance_diff_mat 

            # get all indices where d(a,p) - d(a,n) < 0
            # this constitutes a valid triplet for our model, if the distance between the anchor
            # and the positive is less than the distance between anchor and negative in panphon space
            # we want this relation to be reflected in the embedding space we are learning. any other pair
            # be learning to model the opposite of what we want it to.
            triplet_indices = np.transpose(np.nonzero(valid_triplets < -1e-10)) 

            # create the mask that we can cache for each batch and use during loss comuptation
            mask = np.zeros((batch_size, batch_size, batch_size))
            mask[triplet_indices[:,0], triplet_indices[:,1], triplet_indices[:,2]] = 1
            
            self.batch_triplet_cache[key] = mask

        return self.batch_triplet_cache[key]


    def batch_all_triplet_loss(self, x, data, key, batch_size):
        # get mask for valid triplet indices
        mask = self.get_batch_triplets(data=data, key=key, batch_size=batch_size)
        mask = torch.from_numpy(mask).to(self.model.device)

        # compute distances between each element of batch
        dist_mat = torch.cdist(x, x, p=2)

        # triplet loss computation
        a_to_p_dist = dist_mat.unsqueeze(2)
        a_to_n_dist = dist_mat.unsqueeze(1)
        triplet_loss = a_to_p_dist - a_to_n_dist + self.margin

        # mask out invalid triplets
        triplet_loss = torch.mul(triplet_loss, mask)

        # pass through relu to discard negative values
        triplet_loss = torch.nn.functional.relu(triplet_loss)

        # compute the ratio of positive triplets to valid triplets
        # important metric to track to gauge the progress of the model
        # the ratio of positive triplets going down indicates the model is learning the desired behavior
        # i.e. the loss is going negative d(a,p) - d(a,n) + m < 0 so the positive is closer to anchor than the negative by at least the margin
        num_valid_triplets = torch.sum(mask)
        num_positive_triplets = torch.count_nonzero(triplet_loss)

        # sum the loss matric and average over total # of positive triplets
        triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

    def batch_hard_triplet_loss(self, x, data, key, batch_size):
        # TODO: finish this 
        pass

    def train_step(self, epoch_num):
        self.model.train()
        losses = []
        percent_positive_triplets_list = []
        
        with tqdm(self.train_loader, unit="batch") as train_epoch:
            for batch_idx, data in enumerate(train_epoch):
                train_epoch.set_description(f"Epoch {epoch_num}")
                self.clear_memory()

                ipa = data["ipa"]
                feat_vec = data["feat_vec"]
                batch_size = len(ipa)

                # forward pass
                x = self.model(feat_vec)

                # compute triplet loss
                loss, percent_positive_triplets = self.batch_all_triplet_loss(x=x, data=ipa, key=f'train_{batch_idx}', batch_size=batch_size)

                # step update weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                
                with torch.no_grad():
                    # track the average 2-norm of the embeddings
                    # NOTE: I tried to normalize the output of the model to force all the embeddings to have norm 1 but this really hurt performance for some reason
                    # I saw it was done in other works in computer vision but it did not help for this task
                    embedding_mean_norm = torch.mean(torch.linalg.vector_norm(x, dim=1, ord=2))
                    
                # log stats & update progress bar 
                losses.append(loss.cpu().detach())
                percent_positive_triplets_list.append(percent_positive_triplets.item())
                wandb.log({"train_loss": loss.item(), "percent_positive_triplets": percent_positive_triplets.item(), "embedding_mean_norm": embedding_mean_norm.item()})
                train_epoch.set_postfix(total_loss=loss.item(), percent_positive_triplets=percent_positive_triplets.item(), embedding_mean_norm=embedding_mean_norm.item())

        return losses


    def val_step(self, epoch_num):
        self.model.eval()
        losses = []
        with torch.no_grad():
            with tqdm(self.val_loader, unit="batch") as valid_epoch:
                for batch_idx, data in enumerate(valid_epoch):
                    valid_epoch.set_description(f"Epoch {epoch_num}")
                    self.clear_memory()

                    ipa = data["ipa"]
                    feat_vec = data["feat_vec"]
                    batch_size = len(ipa)

                    # forward pass
                    x = self.model(feat_vec)

                    # compute triplet loss 
                    loss, percent_positive_triplets = self.batch_all_triplet_loss(x=x, data=ipa, key=f'val_{batch_idx}', batch_size=batch_size)
                    
                    # log stats & update progress bar
                    losses.append(loss.cpu().detach())
                    valid_epoch.set_postfix(loss=loss.item(), percent_positive_triplets=percent_positive_triplets.item())
        
        return losses


    def __call__(self):
        for epoch in range(self.n_epochs):
            train_losses = self.train_step(epoch)
            val_losses = self.val_step(epoch)
            wandb.log({"train_loss_avg": np.mean(train_losses),
                       "val_losse_avg": np.mean(val_losses)})
            

            print(
                f"Train loss {np.average(train_losses):8.5f}",
                f"Dev loss {np.average(val_losses):8.5f}"
            )

            if (epoch) % self.eval_every == 0:
                self.model.eval()

                # dev correlations
                dev_pearson, dev_spearman = self.evaluator.compute_corr(self.model, self.data_val, key="dev")
                # train correlations
                train_pearson, train_spearman = self.evaluator.compute_corr(self.model, self.data_train[:500], key="train")
                # dev retrieval
                nn_acc_dict = self.evaluator.compute_nn_precision(self.model, self.data_val, key="dev")
                
                # log wandb stats
                wandb.log({"val_pearson": dev_pearson,
                           "val_spearman": dev_spearman,
                           "train_pearson": train_pearson,
                           "train_spearman": train_spearman,
                           **nn_acc_dict})

                print(
                    f"Dev pearson    {dev_pearson:6.2%}",
                    f"Dev spearman   {dev_spearman:6.2%}",
                    f"Train pearson    {train_pearson:6.2%}",
                    f"Train spearman   {train_spearman:6.2%}"
                )
                
                for k, v in nn_acc_dict.items():
                    print(f'{k}: {v:.2%}')
          
                
        