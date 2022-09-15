import torch.utils.data

from models.vae import *
from util.WIDSMDataLoader import WISDMDataset

dataSet = WISDMDataset("data/wisdm-dataset/raw")

input_dim = 5
hidden_dim = 18
latent_dim = 9

vae = VAE_full(n_features=input_dim, hidden_size=hidden_dim)
dataLoader = torch.utils.data.DataLoader(dataSet)

for x in iter(dataLoader):
    out = vae.forward(x)
    print(out.shape)