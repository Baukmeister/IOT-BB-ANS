from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from torch import lgamma, nn
from torch.distributions import Beta, Normal, Binomial

from loss.vae_loss import VAE_Loss


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_var = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.enc_mu(x)
        log_var = self.enc_log_var(x)
        return mu, torch.exp(log_var)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.D_out = D_out
        self.linear1 = torch.nn.Linear(D_in, H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out * 2)

    def forward(self, z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_linear(x)
        mean, log_var = torch.split(x, self.D_out, dim=1)

        return mean, torch.exp(log_var)


class VAE_full(pl.LightningModule):
    def __init__(self, n_features, batch_size, hidden_size, latent_size, device=None, lr=0.001, wc=0):
        super(VAE_full, self).__init__()
        if self.device is None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.lr = lr
        self.wc = wc

        print(f"Using: {self.device}")
        self.n_features = n_features
        self.encoder = Encoder(D_in=n_features, H=hidden_size, latent_size=latent_size).to(device)
        self.decoder = Decoder(D_in=latent_size, H=hidden_size, D_out=n_features).to(device)
        self.to(device)

    # TODO: Use a learned scale instead of a fixed parameter
    def gaussian_likelihood(self, mean, std, x):

        dist = torch.distributions.Normal(mean, std)

        # measure prob of seeing data under p(x|z)
        log_pxz = dist.log_prob(x)
        # adapt these dimensions
        return log_pxz.sum(1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    # TODO: adapt this to work with pooling
    def plot_prediction(self, prediction_tensors, target_tensors, batch_idx, loss):
        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(projection='3d')

        # select a random subset of the target and predictions to not overcrowd the plot
        predictions = sample(
            [prediction.cpu().detach().numpy() for prediction in prediction_tensors],
            min(500, prediction_tensors.shape[1]) // prediction_tensors.shape[1]
        )
        targets = sample(
            [target.cpu().detach().numpy() for target in target_tensors],
            min(500, target_tensors.shape[1]) // target_tensors.shape[1]
        )

        pred_x = [[pred[i] for i in range(0, len(pred), 3)] for pred in predictions]
        pred_y = [[pred[i] for i in range(1, len(pred), 3)] for pred in predictions]
        pred_z = [[pred[i] for i in range(2, len(pred), 3)] for pred in predictions]

        # plot the points
        pred_ax = ax.scatter(
            [item for sublist in pred_x for item in sublist],
            [item for sublist in pred_y for item in sublist],
            [item for sublist in pred_z for item in sublist]
            , c="blue")

        target_x = [[target[i] for i in range(0, len(target), 3)] for target in targets]
        target_y = [[target[i] for i in range(1, len(target), 3)] for target in targets]
        target_z = [[target[i] for i in range(2, len(target), 3)] for target in targets]

        target_ax = ax.scatter(
            [item for sublist in target_x for item in sublist],
            [item for sublist in target_y for item in sublist],
            [item for sublist in target_z for item in sublist]
            , c="red"
            , alpha=0.3)

        plt.legend([pred_ax, target_ax], ["Predictions", "Targets"])
        plt.title(f"Batch: {batch_idx} - Training loss: {round(loss.item(), 3)}")
        plt.show()

    def reparameterize(self, mu, std):

        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z

    def loss(self, x):

        mu, std = self.encoder(x)
        z = self.reparameterize(mu, std)

        dec_mu, dec_std = self.decoder(z)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(dec_mu, dec_std, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean()
        })

        return elbo

    def forward(self, x_inputs):

        mu, log_var = self.encoder(x_inputs)
        z = self.reparameterize(mu, log_var)

        # decoded
        mean, scale = self.decoder(z)

        return mean, scale

    def training_step(self, batch, batch_idx):

        mean, std = self.forward(batch)
        loss = self.loss(batch)

        if batch_idx % 500 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)

            distribution = torch.distributions.Normal(mean, std)
            outputs = distribution.sample()

            self.plot_prediction(prediction_tensors=outputs, target_tensors=batch, batch_idx=batch_idx,
                                 loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wc)
        return optimizer
