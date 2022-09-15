import torch.utils.data

from models.vae import *
from util.WIDSMDataLoader import WISDMDataset

dataSet = WISDMDataset("data/wisdm-dataset/raw")

input_dim = 5
hidden_dim = 18
latent_dim = 9

vae = VAE_full(n_features=input_dim, hidden_size=hidden_dim)
dataLoader = torch.utils.data.DataLoader(dataSet)

criterion = nn.MSELoss()
optimizer = optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)


for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataLoader, 0):
        # get the inputs; data is a list of [inputs, labels]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vae(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

        # print statistics
        print(f'[{epoch + 1}, {i}] loss: {loss.item()}')
