import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from torch import lgamma
from torch.distributions import Beta, Normal, Binomial

from loss.vae_loss import VAE_Loss


# TODO: investigate the mathematical background of this function
def beta_binomial_log_pdf(k, n, alpha, beta):
    numer = lgamma(n + 1) + lgamma(k + alpha) + lgamma(n - k + beta) + lgamma(alpha + beta)
    denom = lgamma(k + 1) + lgamma(n - k + 1) + lgamma(n + alpha + beta) + lgamma(alpha) + lgamma(beta)
    return numer - denom


# TODO Change this autoencoder to output a distribution (e.g. Beta Binomial) instead of direct samples
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
        self.output_linear = torch.nn.Linear(H, D_out * 2)

    def forward(self, z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_linear(x)
        log_alpha, log_beta = torch.split(x, self.D_out, dim=1)

        return torch.exp(log_alpha), torch.exp(log_beta)


class VAE_full(pl.LightningModule):
    def __init__(self, n_features, batch_size, hidden_size, latent_size, device=None):
        super(VAE_full, self).__init__()
        if self.device is None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.register_buffer('prior_mean', torch.zeros(1))
        self.register_buffer('prior_std', torch.ones(1))
        self.register_buffer('n', torch.ones(batch_size, n_features) * 160.)

        print(f"Using: {self.device}")
        self.n_features = n_features
        self.encoder = Encoder(D_in=n_features, H=hidden_size, latent_size=latent_size).to(device)
        self.decoder = Decoder(D_in=latent_size, H=hidden_size, D_out=n_features).to(device)
        self.to(device)

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

        std = log_var
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, device, epoch, num=64):
        sample = torch.randn(num, self.latent_dim).to(device)
        x_alpha, x_beta = self.decode(sample)
        beta = Beta(x_alpha, x_beta)
        p = beta.sample()
        binomial = Binomial(160, p)
        x_sample = binomial.sample()

        return x_sample

    def loss(self, x):
        # TODO: investigate why this "x" input seems to be full of 0 values
        z_mu, z_std = self.encoder(x.view(-1, self.n_features))
        z = self.reparameterize(z_mu, z_std)  # sample zs

        x_alpha, x_beta = self.decoder(z)
        l = beta_binomial_log_pdf(x.view(-1, self.n_features), self.n,
                                  x_alpha, x_beta)
        l = torch.sum(l, dim=1)
        p_z = torch.sum(Normal(self.prior_mean, self.prior_std).log_prob(z), dim=1)
        q_z = torch.sum(Normal(z_mu, z_std).log_prob(z), dim=1)
        return -torch.mean(l + p_z - q_z) * np.log2(np.e) / self.n_features

    def forward(self, x_inputs):

        mu, log_var = self.encoder(x_inputs)
        z = self.reparameterize(mu, log_var)
        x_alpha, x_beta = self.decoder(z)

        beta_distr = Beta(x_alpha, x_beta)
        p = beta_distr.sample()
        binomial = Binomial(160, p)
        x_reconstructed = binomial.sample().float()
        return x_reconstructed


    def training_step(self, batch, batch_idx):

        outputs = self.forward(batch)
        loss = self.loss(batch)

        if batch_idx % 500 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)
            self.plot_prediction(prediction_tensors=outputs, target_tensors=batch, batch_idx=batch_idx,
                                 loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
