import matplotlib.pyplot as plt
from torch.utils import data

from models.vae import *
from util.WIDSMDataLoader import WISDMDataset


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


def test_model(loss, dataLoader, model):
    targets = []
    preds = []
    losses = []

    for data in dataLoader:
        output, mean, log_var, z = model(data)

        targets.append(data)
        preds.append(output)
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    # plot preds and targets
    plot_prediction(preds, targets)

    # print test loss
    print(f"\nMean test set loss: {mean_loss}")

    return mean_loss



def main():
    dataSet = WISDMDataset("data/wisdm-dataset/raw")

    # CONFIG
    input_dim = 3
    hidden_dim = 32
    latent_dim = 2
    test_set_ratio = 0.001
    train_batch_size = 128
    learning_rate = 0.01
    weight_decay = 0.01

    testSetSize = int(len(dataSet) * test_set_ratio)
    trainSetSize = len(dataSet) - testSetSize
    train_set, test_set = data.random_split(dataSet, [trainSetSize, testSetSize])

    vae = VAE_full(n_features=input_dim, hidden_size=hidden_dim, latent_size=latent_dim)
    trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=8)
    testDataLoader = data.DataLoader(test_set)


    trainer = pl.Trainer(limit_train_batches=100000, max_epochs=1,accelerator='gpu', devices=1)
    trainer.fit(model=vae, train_dataloaders=trainDataLoader)


if __name__ == '__main__':
    main()
