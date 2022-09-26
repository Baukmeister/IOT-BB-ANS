import torch.cuda
from torch import nn, optim

class VAE_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder_linear = nn.Linear(self.input_dim, self.hidden_dim)
        self.fully_connected = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.mean = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim)
        self.log_variance = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.latent_dim)


    def forward(self, x):
        x = self.encoder_linear(x)
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
        x = self.fully_connected(hidden_decoder)
        x = self.fully_connected(hidden_decoder)
        x = self.decoder_linear(x)
        return x


class VAE_full(nn.Module):
  def __init__(self, n_features, hidden_size, latent_size, device=None):
    super(VAE_full, self).__init__()
    self.device = device
    if self.device is None and torch.cuda.is_available():
        self.device = "cuda"
    else:
        self.device = "cpu"

    print(f"Using: {self.device}")
    self.encoder = VAE_encoder(input_dim=n_features, hidden_dim=hidden_size, latent_dim=latent_size).to(self.device)
    self.decoder = VAE_decoder(hidden_dim=hidden_size, latent_dim=latent_size, output_dim=n_features).to(self.device)
    self.to(self.device)

  def reparameterization(self, mean, log_var):
      epsilon = torch.randn_like(log_var).to(self.device)  # sampling epsilon
      z = mean + log_var * epsilon  # reparameterization trick
      return z

  def forward(self, x):

    mean, log_var, x = self.encoder(x)
    z = self.reparameterization(mean, log_var)
    x = self.decoder(z)

    return x, mean, log_var, z


