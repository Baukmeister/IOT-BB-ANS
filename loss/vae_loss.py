import numpy as np
import torch
from torch import nn


class VAE_Loss(torch.nn.Module):
    recon_loss = nn.CrossEntropyLoss(reduction="mean")

    def loss_function(self, input, output, mu, log_var):
        recon_loss = self.recon_loss(input, output)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss

        if loss < 0:
            print("Negative loss. Something went wrong!")

        return loss

    def forward(self, input, output, mu, log_var):
        return self.loss_function(input, output, mu, log_var)
