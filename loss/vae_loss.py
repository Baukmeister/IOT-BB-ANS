import torch
from torch import nn


class VAE_Loss(torch.nn.Module):
    recon_loss = nn.MSELoss()

    def loss_function(self, x, x_hat, mean, log_var):
        reproduction_loss = self.recon_loss(x_hat, x)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        if KLD < 0 or reproduction_loss < 0:
            raise Warning("Negative loss. Something went wrong!")

        return reproduction_loss + KLD

    def forward(self, mean, log_var, x_hat_param, x):

        return self.loss_function(x, x_hat_param, mean, log_var)