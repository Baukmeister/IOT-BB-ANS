import numpy as np
import torch
from matplotlib.patheffects import Normal
from torch import nn

from craystack import Bernoulli


class VAE_Loss(torch.nn.Module):
    recon_loss = nn.CrossEntropyLoss(reduction="mean")

    def loss_function(self, x_input, x_probs, mu, std, p_z, q_z, z):

        #TODO: figure out how to ajdust this to our use_case
        dist = Bernoulli(x_probs)
        l = torch.sum(dist.log_prob(x_input.view(-1, 784)), dim=1)
        p_z = torch.sum(Normal(0, 1).log_prob(z), dim=1)
        q_z = torch.sum(Normal(mu, std).log_prob(z), dim=1)

        loss = -torch.mean(l + p_z - q_z) * np.log2(np.e) / 784.

        if loss < 0:
            print("Negative loss. Something went wrong!")

        return

    def forward(self, x_input, x_probs, mu, std, p_z, q_z, z):
        return self.loss_function(x_input, x_probs, mu, std, p_z, q_z, z)
