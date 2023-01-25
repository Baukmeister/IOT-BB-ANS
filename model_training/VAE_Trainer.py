from datetime import datetime

import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.profilers import SimpleProfiler

from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import VAE_full
from models.vanilla_vae import Vanilla_VAE
from util.experiment_params import Params
from util.io import vae_model_name
import pytorch_lightning as pl
from torch.utils import data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer


class VaeTrainer():
    def __init__(self, params: Params, dataSet: torch.utils.data.Dataset, name, input_dim, grad_clipping=False):

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
        self.range = self.params.range
        self.shift = self.params.shift
        self.model_type = self.params.model_type
        self.metric = self.params.metric

        self.name = name
        self.input_dim = input_dim
        self.gradient_clipping = grad_clipping

        self.dataSet = dataSet

        self.model_name = vae_model_name(
            self.params
        )

        vae_full = VAE_full(
            n_features=self.input_dim,
            range=self.scale_factor,
            batch_size=self.train_batch_size,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            lr=self.learning_rate,
            wc=self.weight_decay,
            plot=False
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
            range=self.range,
            batch_size=self.train_batch_size,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            lr=self.learning_rate,
            wc=self.weight_decay,
            plot=False
        )

        if self.model_type == "full_vae":
            self.model = vae_full
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
        trainDataLoader = data.DataLoader(
            self.train_set,
            batch_size=self.params.train_batch_size,
            shuffle=True,
            drop_last=True
        )
        valDataLoader = data.DataLoader(self.val_set)
        profiler = SimpleProfiler()

        date_time = datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

        wandb_logger = WandbLogger(name=date_time, project="e2e_experiments", group=self.name)
        wandb_logger.watch(self.model, log="all")

        if self.gradient_clipping:
            grad_val = 2.0
        else:
            grad_val = 0


        trainer = pl.Trainer(
            max_epochs=self.params.max_epochs,
            accelerator='gpu',
            devices=1,
            callbacks=[EarlyStopping(monitor="val_loss")],
            profiler=profiler,
            logger=wandb_logger,
            gradient_clip_val=grad_val,
            gradient_clip_algorithm="norm"
        )
        trainer.fit(model=self.model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
        torch.save(self.model.state_dict(), self.model_name)
        wandb.finish()
