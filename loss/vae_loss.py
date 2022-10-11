import numpy as np
import torch
from torch import nn


class VAE_Loss(torch.nn.Module):
    recon_loss = nn.MSELoss()

    def loss_function(self, x_inputs, recons, mu, log_var):
        recons_loss = self.recon_loss(recons, x_inputs)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = recons_loss + 1 * kld_loss

        return loss

    def forward(self, x_inputs, recons, mu, log_var):
        return self.loss_function(x_inputs, recons, mu, log_var)
