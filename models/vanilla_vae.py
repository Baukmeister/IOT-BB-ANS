from random import sample

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple

from torch import nn

from train_VAE import plot_prediction


# Seems like eitehr mean or scale are screwed (not the same for each iteration)
class Vanilla_VAE(pl.LightningModule):
    def __init__(self, n_features, scale_factor, hidden_dims, latent_dim, device=None, lr=0.001, wc=0):

        super(Vanilla_VAE, self).__init__()

        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self.lr = lr
        self.wc = wc
        self.scale_factor = scale_factor
        self.n_features = n_features
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        in_channels = n_features
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=hidden_dims[i],
                              out_features=hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(in_features=hidden_dims[-1],out_features=n_features*2)

        self.to(device)

    def encode(self, input: torch.torch.Tensor) -> List[torch.torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result) * self.scale_factor*100
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        mean, log_var = torch.split(result, self.n_features, dim=1)

        return mean, log_var

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, recons, input, mu, log_var):

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # loss = recons_loss + kld_weight * kld_loss

        loss = recons_loss + 0.0025 * kld_loss
        print(f"KLD: {kld_loss}")
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def forward(self, x_inputs):
        mu, log_var = self.encode(x_inputs)
        z = self.reparameterize(mu, log_var)
        output_mean, output_log_var = self.decode(z)
        distribution = torch.distributions.Normal(output_mean, torch.clamp(torch.exp(output_log_var), 1e-7))
        outputs = distribution.sample()

        return [outputs, x_inputs, mu, log_var]

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def training_step(self, batch, batch_idx):

        recon, x_inputs, mu, log_var = self.forward(batch)
        loss = self.loss(recon, batch, mu, log_var)['loss']

        if batch_idx % 500 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)

            plot_prediction(prediction_tensors=recon, target_tensors=batch, batch_idx=batch_idx,
                                 loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wc)
        return optimizer
