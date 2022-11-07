from torch.utils import data

from models.model_util import plot_prediction
from models.vae import *
from models.vanilla_vae import *
from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name


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
    latent_dim = 15
    val_set_ratio = 0.00
    train_batch_size = 16
    dicretize = True
    learning_rate = 0.001
    weight_decay = 0.01
    scale_factor = 100
    shift = True
    model_type = "full_vae"

    model_name = vae_model_name("./models/trained_models", dicretize, hidden_dim, latent_dim, pooling_factor,
                                scale_factor, model_type, shift)
    dataSet = WISDMDataset("data/wisdm-dataset/raw", pooling_factor=pooling_factor, discretize=dicretize,
                           scaling_factor=scale_factor, shift=True, data_set_size="all")

    valSetSize = int(len(dataSet) * val_set_ratio)
    trainSetSize = len(dataSet) - valSetSize
    train_set, val_set = data.random_split(dataSet, [trainSetSize, valSetSize])

    vae = VAE_full(
        n_features=input_dim,
        scale_factor=scale_factor,
        hidden_size=hidden_dim,
        latent_size=latent_dim,
        lr=learning_rate,
        wc=weight_decay
    )

    vanilla_vae = Vanilla_VAE(
        n_features=input_dim,
        scale_factor=scale_factor,
        hidden_dims=None,
        latent_dim=latent_dim,
        lr=learning_rate,
        wc=weight_decay
    )

    if model_type == "full_vae":
        model = vae
    elif model_type == "vanilla_vae":
        model = vanilla_vae
    else:
        raise ValueError(f"No model defined for '{model_type}'")

    trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=1)
    valDataLoader = data.DataLoader(val_set)

    trainer = pl.Trainer(limit_train_batches=1000000, max_epochs=3, accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()
