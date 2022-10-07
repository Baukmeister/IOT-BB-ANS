import torch.cuda
from matplotlib import pyplot as plt
from torch import nn, optim
import pytorch_lightning as pl
from torch.distributions import Normal

from loss.vae_loss import VAE_Loss
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_std = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.enc_mu(x)
        std = torch.exp(self.enc_std(x))
        return mu, std


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.fc1 = torch.nn.Linear(H, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)
        self.sigmoid = nn.Sigmoid()


    def forward(self, z):
        x = F.relu(self.linear1(z))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.sigmoid(x)
        return output


class VAE_full(pl.LightningModule):
    def __init__(self, n_features, hidden_size, latent_size, device=None):
        super(VAE_full, self).__init__()
        if self.device is None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Using: {self.device}")
        self.encoder = Encoder(D_in=n_features, H=hidden_size, latent_size=latent_size).to(device)
        self.decoder = Decoder(D_in=latent_size, H=hidden_size, D_out=n_features).to(device)
        self.to(device)
        self.loss = VAE_Loss()

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

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):

        mu, std = self.encoder(x)
        z = self.reparameterize(mu, std)
        x_probs = self.decoder(z)
        p_z = torch.sum(Normal(0, 1).log_prob(z), dim=1)
        q_z = torch.sum(Normal(mu, std).log_prob(z), dim=1)
        return x_probs, mu, std, p_z, q_z

    def training_step(self, batch, batch_idx):


        x_probs, mu, std, p_z, q_z  = self.forward(batch)
        loss = self.loss(x_probs, mu, std, p_z, q_z)

        if batch_idx % 1000 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)
            self.plot_prediction(prediction_tensors=x_probs.rsample(), target_tensors=batch, batch_idx=batch_idx, loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
