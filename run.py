import matplotlib.pyplot as plt
import torch.utils.data

from loss.vae_loss import VAE_Loss
from models.vae import *
from util.WIDSMDataLoader import WISDMDataset
from tqdm import tqdm

dataSet = WISDMDataset("data/wisdm-dataset/raw")

# CONFIG
input_dim = 3
hidden_dim = 18
latent_dim = 9
plot = True

vae = VAE_full(n_features=input_dim, hidden_size=hidden_dim, latent_size=latent_dim)
vae_loss = VAE_Loss()
dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=128, shuffle=True)

optimizer = optim.Adam(vae.parameters(), lr=0.001, weight_decay=0.001)

def plot_prediction(prediction, target):
    import matplotlib.pyplot as plt


    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(projection='3d')
    prediction = prediction.cpu().detach().numpy()[0]
    target = target.cpu().detach().numpy()[0]


    # plot the points
    ax.scatter(prediction[0], prediction[1], prediction[2], c="blue")
    ax.scatter(target[0], target[0], target[0], c="red")


    plt.show()

# TODO add test set evalution and store best model
def train():
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        losses = []
        for i, data in enumerate(tqdm(dataLoader)):
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, mean, log_var, z = vae(data)

            loss = vae_loss(mean=mean, log_var=log_var, x_hat_param=output, x=data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if plot and i % 1000 == 0:
                plot_prediction(output, data)
                print(f'[{epoch + 1}, {i}] loss: {loss.item()}')

            if i > 1000:
                break

        plt.title(f"Training loss iteration {i}, epoch {epoch}")
        plt.plot(list(range(len(losses))), losses,)
        plt.show()

if __name__ == '__main__':
    train()