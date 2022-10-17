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


# TODO: investigate the mathematical background of this function
def beta_binomial_log_pdf(k, n, alpha, beta):
    numer = lgamma(n + 1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta)
    denom = lgamma(k + 1) + lgamma(n - k + 1) + lgamma(n + alpha + beta) + lgamma(alpha) + lgamma(beta)
    return numer - denom


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
        log_var = torch.exp(self.enc_log_var(x))
        return mu, log_var


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.D_out = D_out
        self.linear1 = torch.nn.Linear(D_in, H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_linear(x)

        return x


class VAE_full(pl.LightningModule):
    def __init__(self, n_features, batch_size, hidden_size, latent_size, device=None):
        super(VAE_full, self).__init__()
        if self.device is None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        print(f"Using: {self.device}")
        self.n_features = n_features
        self.encoder = Encoder(D_in=n_features, H=hidden_size, latent_size=latent_size).to(device)
        self.decoder = Decoder(D_in=latent_size, H=hidden_size, D_out=n_features).to(device)
        self.to(device)

    # TODO: Use a learned scale instead of a fixed parameter
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        # adapt these dimensions
        return log_pxz.sum()

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

    def plot_prediction(self, prediction_tensors, target_tensors, batch_idx, loss):
        fig = plt.figure(figsize=(4, 4))

        ax = fig.add_subplot(projection='3d')
        predictions = [prediction.cpu().detach().numpy() for prediction in prediction_tensors]
        targets = [target.cpu().detach().numpy() for target in target_tensors]

        # plot the points
        pred_ax = ax.scatter(
            [pred[0] for pred in predictions],
            [pred[1] for pred in predictions],
            [pred[2] for pred in predictions]
            , c="blue")

        target_ax = ax.scatter(
            [target[0] for target in targets],
            [target[1] for target in targets],
            [target[2] for target in targets]
            , c="red"
            , alpha=0.3)

        plt.legend([pred_ax, target_ax], ["Predictions", "Targets"])
        plt.title(f"Batch: {batch_idx} - Training loss: {round(loss.item(), 3)}")
        plt.show()

    def reparameterize(self, mu, log_var):

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z

    def loss(self, x):

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)

        x_hat = self.decoder(z)
        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        std = torch.exp(log_var / 2)
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
        x_hat = self.decoder(z)
        #TODO: This probably need to be sampled (if we want to have samples as outputs)
        return x_hat

    def training_step(self, batch, batch_idx):

        output_probs = self.forward(batch)
        loss = self.loss(batch)

        if batch_idx % 500 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)

            scale = torch.exp(self.log_scale)
            mean = output_probs
            distribution = torch.distributions.Normal(mean, scale)
            outputs = distribution.sample()

            self.plot_prediction(prediction_tensors=outputs, target_tensors=batch, batch_idx=batch_idx,
                                 loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
