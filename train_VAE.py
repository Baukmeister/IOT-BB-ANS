from torch.utils import data

from models.vae import *
from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name


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
    # CONFIG
    pooling_factor = 1
    input_dim = 3 * int(pooling_factor)
    hidden_dim = 32
    latent_dim = 2
    test_set_ratio = 0.001
    train_batch_size = 8
    dicretize = True
    learning_rate = 0.01
    weight_decay = 0.1

    model_name = vae_model_name("./models", dicretize, hidden_dim, latent_dim, pooling_factor)
    dataSet = WISDMDataset("data/wisdm-dataset/raw", pooling_factor=pooling_factor, discretize=dicretize,
                           scaling_factor=1000, data_set_size="single")

    testSetSize = int(len(dataSet) * test_set_ratio)
    trainSetSize = len(dataSet) - testSetSize
    train_set, test_set = data.random_split(dataSet, [trainSetSize, testSetSize])

    vae = VAE_full(
        n_features=input_dim,
        batch_size=train_batch_size,
        hidden_size=hidden_dim,
        latent_size=latent_dim,
        lr=learning_rate,
        wc=weight_decay)
    trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=1)
    testDataLoader = data.DataLoader(test_set)

    trainer = pl.Trainer(limit_train_batches=100000, max_epochs=1, accelerator='gpu', devices=1)
    trainer.fit(model=vae, train_dataloaders=trainDataLoader)
    torch.save(vae.state_dict(), model_name)




if __name__ == '__main__':
    main()
