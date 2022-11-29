from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim, lgamma
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from torch.distributions import Normal, Categorical, Beta, Binomial


from torch import nn

from models.model_util import plot_prediction


def beta_binomial_log_pdf(k, n, alpha, beta):
    numer = lgamma(n+1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta)
    denom = lgamma(k+1) + lgamma(n - k + 1) + lgamma(n + alpha + beta) + lgamma(alpha) + lgamma(beta)
    return numer - denom

class BetaBinomialVAE_sbs(pl.LightningModule):
    def __init__(self, n_features, scale_factor, batch_size, wc=0.0, lr=5e-4, hidden_dim=200, latent_dim=50):
        super(BetaBinomialVAE_sbs, self).__init__()
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.range = float(160 * scale_factor)
        self.batch_size = batch_size
        self.wc = wc
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.register_buffer('n', torch.ones(self.batch_size, n_features) * self.range)

        self.fc1 = nn.Linear(n_features, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)

        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn21 = nn.BatchNorm1d(self.latent_dim)
        self.bn22 = nn.BatchNorm1d(self.latent_dim)

        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, n_features*2)

    def encode(self, x):
        """Return mu, sigma on latent"""
        h = x / self.range  # otherwise we will have numerical issues
        h = F.relu(self.bn1(self.fc1(h)))
        return self.bn21(self.fc21(h)), torch.exp(self.bn22(self.fc22(h)))

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = self.fc4(h)
        log_alpha, log_beta = torch.split(h, self.n_features, dim=1)
        return torch.exp(log_alpha), torch.exp(log_beta)

    def loss(self, x):
        z_mu, z_std = self.encode(x.view(-1, self.n_features))
        z = self.reparameterize(z_mu, z_std)  # sample zs

        x_alpha, x_beta = self.decode(z)
        l = beta_binomial_log_pdf(x.view(-1, self.n_features), self.n,
                                  x_alpha, x_beta)
        l = torch.sum(l, dim=1)
        p_z = torch.sum(Normal(self.prior_mean, self.prior_std).log_prob(z), dim=1)
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * np.log2(np.e) / float(self.n_features)

    def sample(self, device, epoch, num=64):
        sample = torch.randn(num, self.latent_dim).to(device)
        x_alpha, x_beta = self.decode(sample)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(self.range, p)
        x_sample = binomial.sample()
        x_sample = x_sample.float() / float(self.range)
        #TODO: plot image

    def reconstruct(self, x, device):
        x = x.view(-1, self.n_features).float().to(device)
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)  # sample zs
        x_alpha, x_beta = self.decode(z)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(self.range, p)
        x_recon = binomial.sample()
        return x_recon

    def training_step(self, batch, batch_idx):

        if (abs(batch) > self.range).any():
            raise Warning("Batch values are out of range!")


        loss = self.loss(batch)

        if batch_idx % 500 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)

            recon = self.reconstruct(batch, self.device)
            plot_prediction(prediction_tensors=recon, target_tensors=batch, batch_idx=batch_idx,loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wc)
        return optimizer
