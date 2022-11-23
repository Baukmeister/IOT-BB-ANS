from torch.utils import data

from models.vae import *
from models.vanilla_vae import *
from util.SimpleDataLoader import SimpleDataSet


def main():
    # CONFIG
    pooling_factor = 5
    input_dim = 1 * int(pooling_factor)
    hidden_dim = 32
    latent_dim = 1
    val_set_ratio = 0.00
    train_batch_size = 16
    learning_rate = 0.0001
    weight_decay = 0.00001
    model_type = "full_vae"

    model_name = "./models/simple/trained_models/simple_model"
    dataSet = SimpleDataSet(pooling_factor=pooling_factor, data_set_size=int(1e6))
    valSetSize = int(len(dataSet) * val_set_ratio)
    trainSetSize = len(dataSet) - valSetSize
    train_set, val_set = data.random_split(dataSet, [trainSetSize, valSetSize])

    vae = VAE_full(
        n_features=input_dim,
        scale_factor=1,
        hidden_size=hidden_dim,
        latent_size=latent_dim,
        lr=learning_rate,
        wc=weight_decay,
        plot_preds=False
    )

    vanilla_vae = Vanilla_VAE(
        n_features=input_dim,
        scale_factor=1,
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

    trainer = pl.Trainer(limit_train_batches=1000000, max_epochs=2, accelerator='gpu', devices=1)
    trainer.fit(model=model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()
