import os
import time

import numpy as np
import numpy.random
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

from compression import tvae_utils, rans
from compression.rans import stack_depth
from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import VAE_full
from models.vanilla_vae import Vanilla_VAE
from util import bb_util
from util.experiment_params import Params
from util.io import vae_model_name


# TODO rework to offer interface for docker deployment
class NeuralCompressor:

    def __init__(self, params: Params, data_samples: list, input_dim, plot=False):
        self.plot = plot
        self.params = params
        self.state = None

        self.data_samples = data_samples
        self.name = params.model_type
        self.n_features = input_dim

        self.model_name = vae_model_name(params=params)

        self.num_batches = len(self.data_samples) // self.params.compression_batch_size

        vae_full = VAE_full(
            n_features=self.n_features,
            range=self.params.scale_factor,
            batch_size=self.params.train_batch_size,
            hidden_dim=self.params.hidden_dim,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay,
            plot=False
        )

        vanilla_vae = Vanilla_VAE(
            n_features=self.n_features,
            scale_factor=self.params.scale_factor,
            hidden_dims=None,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay
        )

        beta_binomial_vae = BetaBinomialVAE_sbs(
            n_features=self.n_features,
            range=self.params.scale_factor,
            batch_size=self.params.train_batch_size,
            hidden_dim=self.params.hidden_dim,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay,
            plot=False
        )

        if self.params.model_type == "full_vae":
            print("Using Full VAE (gaussian likelihood)")
            self.model = vae_full
            obs_append = tvae_utils.gaussian_obs_append(self.params.range, self.params.obs_precision)
            obs_pop = tvae_utils.gaussian_obs_pop(self.params.range, self.params.obs_precision)
        elif self.params.model_type == "vanilla_vae":
            print("Using Vanilla VAE")
            self.model = vanilla_vae
            # TODO: add this
            obs_append = None
            obs_pop = None
        elif self.params.model_type == "beta_binomial_vae":
            print("Using Beta Binomial VAE (beta-binomial likelihood)")
            self.model = beta_binomial_vae
            obs_append = tvae_utils.beta_binomial_obs_append(self.params.range, self.params.obs_precision)
            obs_pop = tvae_utils.beta_binomial_obs_pop(self.params.range, self.params.obs_precision)
        else:
            raise ValueError(f"No model defined for '{self.params.model_type}'")

        self.model.load_state_dict(torch.load(self.model_name))
        self.model.eval()

        rec_net = tvae_utils.torch_fun_to_numpy_fun(self.model.encode)
        gen_net = tvae_utils.torch_fun_to_numpy_fun(self.model.decode)

        # set up compression methods
        latent_shape = (self.params.compression_batch_size, self.params.latent_dim)

        self.vae_append = bb_util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                                             self.params.prior_precision, self.params.q_precision)
        self.vae_pop = bb_util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                                       self.params.prior_precision, self.params.q_precision)
        self.rng = numpy.random.RandomState()
        self.other_bits = self.rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
        self.state = rans.unflatten(self.other_bits)
        self.stack_sizes = []



    def run_compression(self):

        data_points = self.encode_data_set()
        self.decode_entire_state(data_points)

        if self.plot:
            # plot_stack size:
            x = list(range(len(self.stack_sizes)))
            plt.plot(x, self.stack_sizes)
            plt.title(f"Stack size per per sample")
            plt.show()

    def decode_entire_state(self, data_points):
        decode_start_time = time.time()
        reconstructed_data_points = []
        print("\nDecoding data points ...")
        for _ in tqdm(range(len(data_points))):
            self.remove_from_state(reconstructed_data_points)
        print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))
        recovered_bits = rans.flatten(self.state)
        assert all(self.other_bits == recovered_bits)
        np.testing.assert_equal(reconstructed_data_points, self.data_samples)
        print('\nLossless reconstruction!')

    def encode_data_set(self):
        data_points = np.split(np.reshape(self.data_samples, (len(self.data_samples), -1)), self.num_batches)
        encode_start_time = time.time()
        for i, data_point in tqdm(enumerate(data_points), total=self.params.compression_samples_num):
            self.add_to_state(data_point)
        print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
        compressed_message = rans.flatten(self.state)
        compressed_bits = 32 * (len(compressed_message) - len(self.other_bits))
        compression_rate = compressed_bits / (np.size(data_points) * 32)
        bits_per_datapoint = compressed_bits / np.size(data_points)
        print("Used " + str(compressed_bits) + " bits.")
        print(f'Compression ratio: {round(compression_rate, 4)}. BpD: {bits_per_datapoint}')
        self.state = rans.unflatten(compressed_message)
        return data_points

    def remove_from_state(self, reconstructed_data_points):
        self.state, data_point_ = self.vae_pop(self.state)
        reconstructed_data_points.insert(0, data_point_.numpy()[0])

    def add_to_state(self, data_point):
        self.state = self.vae_append(self.state, data_point)
        self.stack_sizes.append((stack_depth(self.state)))
