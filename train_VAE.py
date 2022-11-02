from torch.utils import data

from models.vae import *
from models.vanilla_vae import *
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
    pooling_factor = 15
    input_dim = 3 * int(pooling_factor)
    hidden_dim = 32
    latent_dim = 2
    test_set_ratio = 0.001
    train_batch_size = 32
    dicretize = True
    learning_rate = 0.000001
    weight_decay = 0.01

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
        wc=weight_decay
    )

    vanilla_vae = Vanilla_VAE(
        n_features=input_dim,
        batch_size=train_batch_size,
        hidden_dims=None,
        latent_dim=latent_dim,
        lr=learning_rate,
        wc=weight_decay
    )

    model = vanilla_vae

    trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=1)
    testDataLoader = data.DataLoader(test_set)

    trainer = pl.Trainer(limit_train_batches=1000000, max_epochs=5, accelerator='gpu', devices=1)
    trainer.fit(model=vanilla_vae, train_dataloaders=trainDataLoader)
    #torch.save(vanilla_vae.state_dict(), model_name)


if __name__ == '__main__':
    main()
