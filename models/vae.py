from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from models.model_util import plot_prediction


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.batch_norm = torch.nn.BatchNorm1d(H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_var = torch.nn.Linear(H, latent_size)
        self.bn_mu = torch.nn.BatchNorm1d(latent_size)
        self.bn_log_var = torch.nn.BatchNorm1d(latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.batch_norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.bn_mu(self.enc_mu(x))
        log_var = self.bn_log_var(self.enc_log_var(x))
        return mu, torch.exp(log_var)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out, scale_factor):
        super(Decoder, self).__init__()
        self.D_out = D_out
        self.linear1 = torch.nn.Linear(D_in, H)
        self.bn = torch.nn.BatchNorm1d(H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.fc4 = torch.nn.Linear(H, H)
        self.fc5 = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out * 2)

    def forward(self, z):
        x = F.relu(self.bn(self.linear1(z)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.output_linear(x)
        mean, log_var = torch.split(x, self.D_out, dim=1)

        return mean, torch.exp(log_var)


# TODO: Figure out why the predictions seem to be very clustered around 0
# Seems like eiter mean or scale are screwed (not the same for each iteration)
class VAE_full(pl.LightningModule):
    def __init__(self, n_features, scale_factor, hidden_size, latent_size, device=None, lr=0.001, wc=0):
        super(VAE_full, self).__init__()
        if self.device is None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.lr = lr
        self.wc = wc
        self.scale_factor = scale_factor

        print(f"Using: {self.device}")
        self.n_features = n_features
        self.encoder = Encoder(D_in=n_features, H=hidden_size, latent_size=latent_size).to(device)
        self.decoder = Decoder(D_in=latent_size, H=hidden_size, D_out=n_features, scale_factor=scale_factor).to(device)
        self.to(device)
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

        return kl.mean()

    def reparameterize(self, mu, std):


        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def mse_loss(self, x, dec_mu, dec_std):
        dec_std = torch.clamp(dec_std, 1e-7)
        distribution = torch.distributions.Normal(dec_mu, dec_std)
        outputs = distribution.sample()
        MSE_fun = torch.nn.MSELoss(reduction='mean')
        MSE_loss = MSE_fun(outputs, x)
        return torch.tensor(MSE_loss, requires_grad=True)

    def loss(self, x):

        mu, std = self.encoder(x)
        z = self.reparameterize(mu, std)

        dec_mu, dec_std = self.decoder(z)

        dist = torch.distributions.Normal(dec_mu, dec_std)
        l = torch.sum(dist.log_prob(x), dim=1)
        p_z = torch.sum(torch.distributions.Normal(0, 1).log_prob(z), dim=1)
        q_z = torch.sum(torch.distributions.Normal(mu, std, 1e-9).log_prob(z), dim=1)

        kl = p_z - q_z

        elbo = -torch.mean(l + p_z - q_z) * np.log2(np.e) / float(x.numel())

        self.log("Elbo Loss", elbo)
        alt_elbo, alt_kl, alt_recon = self.alternate_loss(x)
        return elbo

    def gaussian_likelihood(self, x_hat, scale, x):
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum()
    def alternate_loss(self, x):

        mu, std = self.encoder(x)
        z = self.reparameterize(mu, std)

        dec_mu, dec_std = self.decoder(z)

        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        distribution = torch.distributions.Normal(dec_mu, torch.clamp(dec_std, 1e-7))
        x_hat = distribution.sample()

        # reconstruction loss
        log_scale = torch.Tensor([0.0]).to(self.device)
        recon_loss = self.gaussian_likelihood(dec_mu, dec_std, x)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        return elbo, kl, recon_loss

    def forward(self, x_inputs):

        mu, std = self.encoder(x_inputs)
        z = self.reparameterize(mu, std)

        dec_mu, dec_std = self.decoder(z)

        return dec_mu, dec_std

    def training_step(self, batch, batch_idx):

        mean, std = self.forward(batch)
        loss = self.loss(batch)

        if batch_idx % 500 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', round(loss.item(), 5))

            distribution = torch.distributions.Normal(mean, torch.clamp(std, 1e-7))
            outputs = distribution.sample()

            plot_prediction(prediction_tensors=outputs, target_tensors=batch, batch_idx=batch_idx,
                            loss=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        mean, log_var = self.forward(batch)
        loss = self.loss(batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': log}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wc)
        return optimizer
