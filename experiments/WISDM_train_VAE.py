from pytorch_lightning.profilers import SimpleProfiler
from torch.utils import data

from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import *
from models.vanilla_vae import *
from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name


def main():
    # CONFIG
    pooling_factor = 10
    input_dim = 3 * int(pooling_factor)
    hidden_dim = 200
    latent_dim = 25
    val_set_ratio = 0.00
    train_batch_size = 32
    dicretize = True
    learning_rate = 0.005
    weight_decay = 0.0001
    scale_factor = 10
    shift = True
    model_type = "beta_binomial_vae"
    data_set_type = "accel"

    model_name = vae_model_name("./models/trained_models", dicretize, hidden_dim, latent_dim, pooling_factor,
                                scale_factor, model_type, shift, data_set_type)
    dataSet = WISDMDataset("data/wisdm-dataset/raw", pooling_factor=pooling_factor, discretize=dicretize,
                           scaling_factor=scale_factor, shift=shift, data_set_size=data_set_type, caching=False)

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

    beta_binomial_vae = BetaBinomialVAE_sbs(
        n_features=input_dim,
        scale_factor=scale_factor,
        batch_size=train_batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=learning_rate,
        wc=weight_decay
    )

    if model_type == "full_vae":
        model = vae
    elif model_type == "vanilla_vae":
        model = vanilla_vae
    elif model_type == "beta_binomial_vae":
        model = beta_binomial_vae
    else:
        raise ValueError(f"No model defined for '{model_type}'")

    trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=1)
    valDataLoader = data.DataLoader(val_set)

    profiler = SimpleProfiler()
    # profiler = PyTorchProfiler()

    trainer = pl.Trainer(limit_train_batches=10000, max_epochs=10, accelerator='gpu', devices=1, profiler=profiler)
    trainer.fit(model=model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()
