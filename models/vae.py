import torch.cuda
from torch import nn


class VAE_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_encoder, self).__init__()

        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, 2)
        self.fully_connected = nn.Linear(hidden_dim,hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0,2)


    def forward(self, x):
        h_ = self.LeakyReLU(self.encoder_lstm(x))
        h_ = self.LeakyReLU(self.fully_connected(h_))
        mean = self.mean(h_)
        log_var = self.log_variance(h_)

        return mean, log_var, h_

class VAE_decoder(nn.Module):

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAE_decoder, self).__init__()


        self.gru_dec1 = nn.LSTM(latent_dim, latent_dim,
                               batch_first=True, dropout=0,
                               bidirectional=False)

        self.gru_dec2 = nn.LSTM(latent_dim, hidden_dim,
                               batch_first=True, dropout=0,
                               bidirectional=True)

        self.output_layer = nn.Linear(self.hidden_size * 2, output_dim, bias=True)
        self.act = nn.ReLU()

    def forward(self, x, seq_len):
        x = x.repeat(1, seq_len, 1)
        x, _ = self.gru_dec1(x)
        x, _ = self.gru_dec2(x)
        return self.act(self.output_layer(x))


class VAE_full(nn.Module):
  def __init__(self, n_features, latent_dim , hidden_size):
    super(VAE_full, self).__init__()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    self.encoder = VAE_encoder(n_features, latent_dim, hidden_size).to(device)
    self.decoder = VAE_decoder(n_features, latent_dim, hidden_size).to(device)

  def forward(self, x):
    mean, log_var, h_ = self.encoder(x)
    out = self.decoder(h_)
    return out