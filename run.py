import torch.utils.data

from models.vae import VAE
from util.WIDSMDataLoader import WISDMDataset

dataSet = WISDMDataset("data/wisdm-dataset/raw")

vae = VAE()
dataLoader = torch.utils.data.DataLoader(dataSet)

for x in iter(dataLoader):
    print(vae.forward(x))