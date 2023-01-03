from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils import data

from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import *
from models.vanilla_vae import *
from util.IntelLabDataLoader import IntelLabDataset
from util.io import vae_model_name
from pytorch_lightning import loggers as pl_loggers



def main():
    # CONFIG
    pooling_factor = 10
    hidden_dim = 50
    latent_dim = 10
    train_set_ratio = 1.0
    val_set_ratio = 0.01
    train_batch_size = 8
    dicretize = True
    learning_rate = 0.0001
    weight_decay = 0.01
    scale_factor = 109
    shift = True
    model_type = "beta_binomial_vae"
    metric = "temperature"

    if metric == "all":
        input_dim = 4 * int(pooling_factor)
    else:
        input_dim = 1 * int(pooling_factor)

    model_name = vae_model_name(
        "../models/trained_models/IntelLab",
        dicretize,
        hidden_dim,
        latent_dim,
        pooling_factor,
        scale_factor,
        model_type,
        shift,
        data_set_type=metric
    )

    dataSet = IntelLabDataset(
        "../data/IntelLabData",
        pooling_factor=pooling_factor,
        scaling_factor=scale_factor,
        caching=False,
        metric=metric
    )

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
        range=dataSet.range,
        batch_size=train_batch_size,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lr=learning_rate,
        wc=weight_decay,
        plot=False
    )

    if model_type == "full_vae":
        model = vae
    elif model_type == "vanilla_vae":
        model = vanilla_vae
    elif model_type == "beta_binomial_vae":
        model = beta_binomial_vae
    else:
        raise ValueError(f"No model defined for '{model_type}'")

    trainDataLoader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=8)
    valDataLoader = data.DataLoader(val_set)

    profiler = SimpleProfiler()
    # profiler = PyTorchProfiler()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="intel_lab_logs/")


    trainer = pl.Trainer(
        limit_train_batches=int((train_set_ratio * trainSetSize) / train_batch_size),
        max_epochs=1,
        accelerator='gpu',
        devices=1,
        callbacks=[EarlyStopping(monitor="val_loss")],
        profiler=profiler,
        logger=tb_logger
    )
    trainer.fit(model=model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
    torch.save(model.state_dict(), model_name)


if __name__ == '__main__':
    main()
