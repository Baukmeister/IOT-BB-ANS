import torch.utils.data

from models.vae import *
from util.WIDSMDataLoader import WISDMDataset

dataSet = WISDMDataset("data/wisdm-dataset/raw")

input_dim = 5
hidden_dim = 18
latent_dim = 9

vae_encode = VAE_encoder(input_dim = input_dim, hidden_dim = hidden_dim)
vae_decode = VAE_decoder(latent_dim = latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
vae = VAE_full()
dataLoader = torch.utils.data.DataLoader(dataSet)

for x in iter(dataLoader):
    print(vae.forward(x))