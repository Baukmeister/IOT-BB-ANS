import matplotlib.pyplot as plt
import torch.utils.data

from loss.vae_loss import VAE_Loss
from models.vae import *
from util.WIDSMDataLoader import WISDMDataset
from tqdm import tqdm
from torch.utils import data

dataSet = WISDMDataset("data/wisdm-dataset/raw")

# CONFIG
input_dim = 3
hidden_dim = 18
latent_dim = 2
plot = True
test_set_ratio = 0.001
train_batch_size = 128
learning_rate = 0.001
weight_decay = 0.001
plot_epoch_interval = 2000

testSetSize = int(len(dataSet) * test_set_ratio)
trainSetSize = len(dataSet) - testSetSize
train_set, test_set = data.random_split(dataSet, [trainSetSize, testSetSize])

vae = VAE_full(n_features=input_dim, hidden_size=hidden_dim, latent_size=latent_dim)
vae_loss = VAE_Loss()
trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
testDataLoader = data.DataLoader(test_set)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=weight_decay)


def plot_prediction(prediction_tensors, target_tensors):
    fig = plt.figure(figsize=(4, 4))

    ax = fig.add_subplot(projection='3d')
    predictions = [prediction.cpu().detach().numpy() for prediction in prediction_tensors]
    targets = [target.cpu().detach().numpy() for target in target_tensors]

    # plot the points
    pred_ax = ax.scatter(
        [pred[0][0] for pred in predictions],
        [pred[0][1] for pred in predictions],
        [pred[0][2] for pred in predictions]
        , c="blue")

    target_ax = ax.scatter(
        [target[0][0] for target in targets],
        [target[0][1] for target in targets],
        [target[0][2] for target in targets]
        , c="red"
        , alpha=0.3)

    plt.legend([pred_ax, target_ax], ["Predictions", "Targets"])

    plt.show()


def test_model():
    targets = []
    preds = []
    losses = []

    for data in testDataLoader:
        output, mean, log_var, z = vae(data)
        loss = vae_loss(mean=mean, log_var=log_var, x_hat_param=output, x=data)

        targets.append(data)
        preds.append(output)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    # plot preds and targets
    plot_prediction(preds, targets)

    # print test loss
    print(f"\nMean test set loss: {mean_loss}")

    return mean_loss

def train():
    for epoch in range(1):  # loop over the dataset multiple times

        best_loss = float("inf")
        train_losses = []
        test_losses = []
        for i, data in enumerate(tqdm(trainDataLoader)):
            # get the inputs; data is a list of [inputs, labels]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output, mean, log_var, z = vae(data)

            loss = vae_loss(mean=mean, log_var=log_var, x_hat_param=output, x=data)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if plot and i % plot_epoch_interval == 0:
                print(f'\n[epoch: {epoch + 1}, batch: {i}]\ntraining loss: {loss.item()}')
                test_loss = test_model()
                test_losses.append(test_loss)

                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(vae.state_dict(), f"models/trained_vae_l{latent_dim}_h{hidden_dim}")
                    print("\nStored model as new best model")

        plt.title(f"Training loss iteration epoch {epoch + 1}")
        plt.plot(list(range(len(train_losses))), train_losses, )
        plt.show()

        plt.title(f"Test loss iteration epoch {epoch + 1}")
        plt.plot(list(range(len(test_losses))), test_losses)
        plt.show()


if __name__ == '__main__':
    train()
