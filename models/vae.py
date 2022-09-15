import torch.cuda
from torch import nn, optim


class VAE_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(VAE_encoder, self).__init__()

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, 2)
        self.fully_connected = nn.Linear(hidden_dim,hidden_dim)

        self.LeakyReLU = nn.LeakyReLU(0,2)


    def forward(self, x):
        x, _ = self.encoder_lstm(x)
        x = self.fully_connected(x)

        return x

class VAE_decoder(nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(VAE_decoder, self).__init__()

        self.fully_connected = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, output_dim, 2)

    def forward(self, x):
        x = self.fully_connected(x)
        x = self.decoder_lstm(x)
        return x


class VAE_full(nn.Module):
  def __init__(self, n_features, hidden_size):
    super(VAE_full, self).__init__()

    if torch.cuda.is_available():
        device = "cpu"
    else:
        device = "cpu"

    self.encoder = VAE_encoder(input_dim=n_features, hidden_dim=hidden_size).to(device)
    self.decoder = VAE_decoder(hidden_dim=hidden_size, output_dim=n_features).to(device)

  def forward(self, x):

    input = x.detach().numpy()

    x = self.encoder(x)
    x, _ = self.decoder(x)

    output = x.detach().numpy()
    return x


