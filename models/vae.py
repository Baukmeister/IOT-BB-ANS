import torch.cuda
from torch import nn, optim

#TODO: rework model using this repo https://github.com/Khamies/LSTM-Variational-AutoEncoder/blob/main/model.py

class VAE_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(VAE_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.encoder_lstm = nn.LSTM(self.input_dim, self.hidden_dim, 2)
        self.fully_connected = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.mean = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim)
        self.log_variance = torch.nn.Linear(in_features= self.hidden_dim, out_features=self.latent_dim)


    def forward(self, x):
        x, _ = self.encoder_lstm(x)
        x = self.fully_connected(x)
        mean = self.mean(x)
        log_var = self.log_variance(x)
        std = torch.exp(0.5 * log_var)  # e^(0.5 log_var) = var^0.5

        # Generate a unit gaussian noise.
        batch_size = x.size(0)
        seq_len = x.size(1)
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)

        z = noise * std + mean

        return z, mean, log_var, x

class VAE_decoder(nn.Module):

    def __init__(self, hidden_dim, latent_dim, output_dim):
        super(VAE_decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.init_hidden_decoder = torch.nn.Linear(in_features=self.latent_dim,
                                                   out_features=self.hidden_dim)

        self.fully_connected = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, output_dim, 2)

    def forward(self, z):

        hidden_decoder = self.init_hidden_decoder(z)

        x = self.fully_connected(hidden_decoder)
        x = self.decoder_lstm(x)
        return x


class VAE_full(nn.Module):
  def __init__(self, n_features, hidden_size, latent_size):
    super(VAE_full, self).__init__()

    if torch.cuda.is_available():
        self.device = "cpu"
    else:
        self.device = "cpu"

    self.encoder = VAE_encoder(input_dim=n_features, hidden_dim=hidden_size, latent_dim=latent_size, device=self.device).to(self.device)
    self.decoder = VAE_decoder(hidden_dim=hidden_size, latent_dim=latent_size, output_dim=n_features).to(self.device)

  def forward(self, x):

    input = x.detach().numpy()

    z, mean, log_var, x = self.encoder(x)
    x, _ = self.decoder(z)

    output = x.detach().numpy()
    return x, mean, log_var, z


