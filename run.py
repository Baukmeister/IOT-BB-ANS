import torch.utils.data

from loss.vae_loss import VAE_Loss
from models.vae import *
from util.WIDSMDataLoader import WISDMDataset

dataSet = WISDMDataset("data/wisdm-dataset/raw")

input_dim = 5
hidden_dim = 18
latent_dim = 9

vae = VAE_full(n_features=input_dim, hidden_size=hidden_dim, latent_size=latent_dim)
vae_loss = VAE_Loss()
dataLoader = torch.utils.data.DataLoader(dataSet)

criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)


for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output, mean, log_var, z = vae(data)
        mloss, KL_loss, recon_loss = vae_loss(mu=mean, log_var=log_var, z=z, x_hat_param=output, x=data)
        mloss.backward()
        optimizer.step()

        # print statistics
        print(f'[{epoch + 1}, {i}] loss: {mloss.item()}')
