from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.lstmInput = nn.LSTM(6,18,2)

    def forward(self, x):

        x = self.lstmInput(x)
        return x