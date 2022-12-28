from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import VAE_full
from models.vanilla_vae import Vanilla_VAE
from util.io import vae_model_name
from torch.utils import data


class VaeTrainer():
    def __init__(self, params, name, input_dim):

        # extract params
        self.pooling_factor = params.pooling_factor
        self.hidden_dim = params.hidden_dim
        self.latent_dim = params.latent_dim
        self.train_set_ratio = params.train_set_ratio
        self.val_set_ratio = params.val_set_ratio
        self.train_batch_size = params.train_batch_size
        self.dicretize = params.discretize
        self.learning_rate = params.learning_rate
        self.weight_decay = params.weight_decay
        self.scale_factor = params.scale_factor
        self.shift = params.shift
        self.model_type = params.model_type
        self.metric = params.metric

        self.name = name
        self.input_dim = input_dim

        self.model_name = vae_model_name(
            f"../models/trained_models/{self.name}",
            self.dicretize,
            self.hidden_dim,
            self.latent_dim,
            self.pooling_factor,
            self.scale_factor,
            self.model_type,
            self.shift,
            data_set_type=self.metric
        )

        model_name = "../models/simple/trained_models/simple_model"

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


    def train_model(self):
        raise NotImplementedError
