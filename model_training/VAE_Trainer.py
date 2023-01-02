import torch
from pytorch_lightning.profilers import SimpleProfiler

from experiment_pipelines.experiment_params import Params
from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import VAE_full
from models.vanilla_vae import Vanilla_VAE
from util.io import vae_model_name
import pytorch_lightning as pl
from torch.utils import data


class VaeTrainer():
    def __init__(self, params : Params, dataSet: torch.utils.data.Dataset, name, input_dim):

        # extract params
        self.params = params
        self.pooling_factor = self.params.pooling_factor
        self.hidden_dim = self.params.hidden_dim
        self.latent_dim = self.params.latent_dim
        self.train_set_ratio = self.params.train_set_ratio
        self.val_set_ratio = self.params.val_set_ratio
        self.train_batch_size = self.params.train_batch_size
        self.discretize = self.params.discretize
        self.learning_rate = self.params.learning_rate
        self.weight_decay = self.params.weight_decay
        self.scale_factor = self.params.scale_factor
        self.shift = self.params.shift
        self.model_type = self.params.model_type
        self.metric = self.params.metric

        self.name = name
        self.input_dim = input_dim

        self.dataSet = dataSet

        self.model_name = vae_model_name(
            f"../models/trained_models/{self.name}",
            self.discretize,
            self.hidden_dim,
            self.latent_dim,
            self.pooling_factor,
            self.scale_factor,
            self.model_type,
            self.shift,
            data_set_type=self.metric
        )


        vae = VAE_full(
            n_features=self.input_dim,
            scale_factor=self.scale_factor,
            hidden_size=self.hidden_dim,
            latent_size=self.latent_dim,
            lr=self.learning_rate,
            wc=self.weight_decay
        )

        vanilla_vae = Vanilla_VAE(
            n_features=self.input_dim,
            scale_factor=self.scale_factor,
            hidden_dims=None,
            latent_dim=self.latent_dim,
            lr=self.learning_rate,
            wc=self.weight_decay
        )

        beta_binomial_vae = BetaBinomialVAE_sbs(
            n_features=self.input_dim,
            range=self.scale_factor,
            batch_size=self.train_batch_size,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            lr=self.learning_rate,
            wc=self.weight_decay,
            plot=False
        )

        if self.model_type == "full_vae":
            self.model = vae
        elif self.model_type == "vanilla_vae":
            self.model = vanilla_vae
        elif self.model_type == "beta_binomial_vae":
            self.model = beta_binomial_vae
        else:
            raise ValueError(f"No model defined for '{self.model_type}'")


        self.valSetSize = int(len(self.dataSet) * self.val_set_ratio)
        self.trainSetSize = len(self.dataSet) - self.valSetSize
        self.train_set, self.val_set = data.random_split(self.dataSet, [self.trainSetSize, self.valSetSize])

    def train_model(self):
        trainDataLoader = data.DataLoader(self.train_set, batch_size=self.params.train_batch_size,  shuffle=True,
                                          drop_last=True)
        valDataLoader = data.DataLoader(self.val_set)
        profiler = SimpleProfiler()
        trainer = pl.Trainer(limit_train_batches=self.params.train_batches, max_epochs=self.params.max_epochs, accelerator='gpu', devices=1,
                             profiler=profiler)
        trainer.fit(model=self.model, train_dataloaders= trainDataLoader, val_dataloaders= valDataLoader,)
        torch.save(self.model.state_dict(), self.model_name)
