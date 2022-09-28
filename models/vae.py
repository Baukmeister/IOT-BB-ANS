import torch.cuda
from matplotlib import pyplot as plt
from torch import nn, optim
import pytorch_lightning as pl

from loss.vae_loss import VAE_Loss


class VAE_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.fully_connected = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim)
        self.log_variance = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim)

    def forward(self, x):
        x = self.encoder_linear(x)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        mean = self.mean(x)
        log_var = self.log_variance(x)

        return mean, log_var, x


class VAE_decoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, output_dim):
        super(VAE_decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_dim,
                                                   out_features=self.hidden_dim)

        self.fully_connected = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        hidden_decoder = self.init_hidden_decoder(z)

        x = self.fully_connected(hidden_decoder)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        x = self.fully_connected(x)
        x = self.decoder_linear(x)
        return x


class VAE_full(pl.LightningModule):
    def __init__(self, n_features, hidden_size, latent_size, device=None):
        super(VAE_full, self).__init__()
        if self.device is None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Using: {self.device}")
        self.encoder = VAE_encoder(input_dim=n_features, hidden_dim=hidden_size, latent_dim=latent_size).to(device)
        self.decoder = VAE_decoder(hidden_dim=hidden_size, latent_dim=latent_size, output_dim=n_features).to(device)
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
        plt.title(f"Batch: {batch_idx} - Training loss: {loss.item()}")
        plt.show()

    def reparameterization(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(self.device)  # sampling epsilon
        z = mean + log_var * epsilon  # reparameterization trick
        return z

    def forward(self, x):

        mean, log_var, x = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        x = self.decoder(z)

        return x, mean, log_var, z

    def training_step(self, batch, batch_idx):

        output, mean, log_var, z = self.forward(batch)
        loss = self.loss(mean=mean, log_var=log_var, x_hat_param=output, x=batch)

        if batch_idx % 3000 == 0:
            self.log(f'\n[batch: {batch_idx}]\ntraining loss', loss)
            self.plot_prediction(prediction_tensors=output, target_tensors=batch, batch_idx=batch_idx, loss=loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer