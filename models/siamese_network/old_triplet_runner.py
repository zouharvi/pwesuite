from tqdm import tqdm
import torch
import numpy as np
from multiprocessing.pool import ThreadPool
import random
import gc
import panphon2
import wandb
from util import get_embeddings, seed_everything

class TripletRunner:
    def __init__(self, model, criterion, optimizer, data_train, data_val, train_loader, val_loader, evaluator, n_epochs, wandb_name="", wandb_entity="", eval_every=5):
        seed_everything(42)

        self.model = model

        # loss stuff and optimizer
        self.triplet_loss_fn = criterion
        self.margin = self.triplet_loss_fn.margin
        self.mse_loss_fn = torch.nn.MSELoss()
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


    def compute_batch_panphon_distance(self, data, key):
        def compute_panphon_dist(y, data):
            fed = panphon2.FeatureTable().feature_edit_distance
            return [fed(x, y) for x in data]

        if key not in self.panphon_distance_cache:
            with ThreadPool() as pool:
                distances = pool.map(lambda y: compute_panphon_dist(y, data), data)
            distances = np.array(distances)
            self.panphon_distance_cache[key] = distances
            return distances
        else:
            return self.panphon_distance_cache[key]
        

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
            panphon_distances = self.compute_batch_panphon_distance(data=data, key=key)

            # # get matrix of all possible distances d(a,p) - d(a,n) + margin
            distance_diff_mat = np.expand_dims(panphon_distances, 2) - np.expand_dims(panphon_distances, 1)
            # mask distance matrix 
            valid_triplets = distinct_mask * distance_diff_mat # I wasn't applying the distinct mask to the difference matrix

            # get all indices where d(a,p) - d(a,n) < 0
            # this constitutes a valid triplet for our model, if the distance between the anchor
            # and the positive is less than the distance between anchor and negative in panphon space
            # we want this relation to be reflected in the embedding space we are learning. any other pair
            # be learning to model the opposite of what we want it to.
            triplet_indices = np.transpose(np.nonzero(valid_triplets < -1e-10)) 
            
            self.batch_triplet_cache[key] = triplet_indices
        else:
            # triplet_indices = self.batch_triplet_cache[key]
            triplet_indices = self.batch_triplet_cache[key]

        # mask = self.batch_triplet_cache[key]

        return triplet_indices
        # return torch.from_numpy(mask)


    def semi_hard_negative_mine(self, anchor, positive, negative):
        # semi-hard negative mining: seek to find all triplets that satisfy
        # d(a,p) < d(a,n) < d(a,p) + margin  ==> d(a,p) - d(a,n) + margin > 0
        # i.e. the triplets where the negative is further away from the anchor
        # than the positive, but where the loss is still positive
        a_to_p_dist = torch.norm(anchor - positive, p=2, dim=1)
        a_to_n_dist = torch.norm(anchor - negative, p=2, dim=1)
        mask = (a_to_p_dist < a_to_n_dist) & (a_to_n_dist < a_to_p_dist + self.margin)
        return anchor[mask], positive[mask], negative[mask]


    def hard_negative_mine(self, anchor, positive, negative):
        # hard negative mining: triplets where the negative is closer to the 
        # anchor than the positive is
        # d(a,n) < d(a,p)
        a_to_p_dist = torch.norm(anchor - positive, p=2, dim=1)
        a_to_n_dist = torch.norm(anchor - negative, p=2, dim=1)
        mask = a_to_n_dist < a_to_p_dist + self.margin
        return anchor[mask], positive[mask], negative[mask]


    def compute_mse_loss(self, ipa, fv1_emb):
        # randomly select data
        data = random.choices(self.data_train, k=len(ipa))
        ipa2 = [d[0] for d in data]
        feat_vec2 = [d[1] for d in data]

        dists_true = torch.Tensor([self.fed(w1, w2) for w1, w2 in zip(ipa, ipa2)]).to(self.model.device)
        fv2_emb = self.model(feat_vec2)

        dists_hyp = self.dists_emb(fv1_emb, fv2_emb)

        loss = self.mse_loss_fn(dists_hyp, dists_true)

        return loss

    def batch_all_triplet_loss(self, x, data, key, batch_size):
        # create mask for valid triplets
        triplet_indices = self.get_batch_triplets(data=data, key=key, batch_size=batch_size)

        mask = np.zeros((batch_size, batch_size, batch_size))
        mask[triplet_indices[:,0], triplet_indices[:,1], triplet_indices[:,2]] = 1
        mask = torch.from_numpy(mask).to(self.model.device)

        # mask = self.get_batch_triplets(data=data, key=key, batch_size=batch_size).to(self.model.device)

        dist_mat = torch.cdist(x, x, p=2)
        a_to_p_dist = dist_mat.unsqueeze(2)
        a_to_n_dist = dist_mat.unsqueeze(1)

        triplet_loss = a_to_p_dist - a_to_n_dist + self.margin
        triplet_loss = torch.mul(triplet_loss, mask)
        triplet_loss = torch.nn.functional.relu(triplet_loss)

        num_valid_triplets = torch.sum(mask)
        num_positive_triplets = torch.count_nonzero(triplet_loss)

        triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)
        fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

        return triplet_loss, fraction_positive_triplets

    def batch_hard_triplet_loss(self, x, data, key, batch_size):
        # TODO: finish this 
        pass

    def train_step(self, epoch_num):
        self.model.train()
        # self.model.turn_proj_head_on()
        losses = []
        percent_positive_triplets_list = []
        triplet_losses = []
        
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
                # self.scheduler.step()

                # log stats & update progress bar 
                with torch.no_grad():
                    embedding_mean_norm = torch.mean(torch.linalg.vector_norm(x, dim=1, ord=2))
                    
                losses.append(loss.cpu().detach())
                percent_positive_triplets_list.append(percent_positive_triplets.item())
                wandb.log({"train_loss": loss.item(), "percent_positive_triplets": percent_positive_triplets.item(), "embedding_mean_norm": embedding_mean_norm.item()})
                train_epoch.set_postfix(total_loss=loss.item(), percent_positive_triplets=percent_positive_triplets.item(), embedding_mean_norm=embedding_mean_norm.item())

        return losses


    def val_step(self, epoch_num):
        self.model.eval()
        # self.model.turn_proj_head_on()
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
                # turn proj head off and set model on eval mode
                # self.model.turn_proj_head_off()
                self.model.eval()

                # dev correlations
                dev_pearson, dev_spearman = self.evaluator.compute_corr(self.model, self.data_val, key="dev")
                # train correlations
                train_pearson, train_spearman = self.evaluator.compute_corr(self.model, self.data_train[:500], key="train")
                # dev correlations
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

            # self.scheduler.step()
          
                
        