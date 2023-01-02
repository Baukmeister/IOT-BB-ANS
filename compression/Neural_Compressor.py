import os
import time

import numpy as np
import numpy.random
import torch.utils.data
from tqdm import tqdm

from compression import tvae_utils, rans
from experiment_pipelines.experiment_params import Params
from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import VAE_full
from models.vanilla_vae import Vanilla_VAE
from util import bb_util
from util.io import vae_model_name


class NeuralCompressor():

    def __init__(self, params: Params, dataSet: torch.utils.data.Dataset, name: str):
        self.params = params

        self.dataSet = dataSet
        self.name = name

        self.model_name = vae_model_name(
            f"../models/trained_models/{self.name}",
            self.params.discretize,
            self.params.hidden_dim,
            self.params.latent_dim,
            self.params.pooling_factor,
            self.params.scale_factor,
            self.params.model_type,
            self.params.shift,
            data_set_type =self.params.metric
        )

        self.data_points_singles = [self.dataSet.__getitem__(i).cpu().numpy() for i in range(self.params.compression_samples_num)]
        self.num_batches = len(self.data_points_singles) // self.params.compression_batch_size

        vae = VAE_full(
            n_features=self.params.input_dim,
            scale_factor=self.params.scale_factor,
            hidden_size=self.params.hidden_dim,
            latent_size=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay
        )

        vanilla_vae = Vanilla_VAE(
            n_features=self.params.input_dim,
            scale_factor=self.params.scale_factor,
            hidden_dims=None,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay
        )

        beta_binomial_vae = BetaBinomialVAE_sbs(
            n_features=self.params.input_dim,
            range=self.params.scale_factor,
            batch_size=self.params.train_batch_size,
            hidden_dim=self.params.hidden_dim,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay,
            plot=False
        )

        if self.params.model_type == "full_vae":
            self.model = vae
        elif self.params.model_type == "vanilla_vae":
            self.model = vanilla_vae
        elif self.params.model_type == "beta_binomial_vae":
            self.model = beta_binomial_vae
        else:
            raise ValueError(f"No model defined for '{self.params.model_type}'")

        self.model.load_state_dict(torch.load(self.model_name))
        rec_net = tvae_utils.torch_fun_to_numpy_fun(self.model.encode)
        gen_net = tvae_utils.torch_fun_to_numpy_fun(self.model.decode)

        # set up compression methods
        latent_shape = (self.params.compression_batch_size, self.params.latent_dim)
        latent_size = np.prod(latent_shape)
        obs_shape = (self.params.compression_batch_size, self.params.input_dim * int(self.params.pooling_factor))
        obs_size = np.prod(obs_shape)

        obs_append = tvae_utils.beta_binomial_obs_append(self.params.scale_factor, self.params.obs_precision)
        obs_pop = tvae_utils.beta_binomial_obs_pop(self.params.scale_factor, self.params.obs_precision)

        self.vae_append = bb_util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                                        self.params.prior_precision, self.params.q_precision)
        self.vae_pop = bb_util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                                  self.params.prior_precision, self.params.q_precision)

    def run_compression(self):
        rng = numpy.random.RandomState()

        other_bits = rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
        state = rans.unflatten(other_bits)
        data_points = np.split(np.reshape(self.data_points_singles, (len(self.data_points_singles), -1)), self.num_batches)

        compress_lengths = []

        print_interval = 100
        encode_start_time = time.time()
        for i, data_point in enumerate(data_points):
            state = self.vae_append(state, data_point)

            if not i % print_interval:
                print('Encoded {}'.format(i))

            compressed_length = 32 * (len(rans.flatten(state)) - len(other_bits)) / (i + 1)
            compress_lengths.append(compressed_length)

        print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
        compressed_message = rans.flatten(state)

        compressed_bits = 32 * (len(compressed_message) - len(other_bits))
        print("Used " + str(compressed_bits) + " bits.")
        print(f'This is {compressed_bits / np.size(data_points)} bits per data point')

        if not os.path.exists('results'):
            os.mkdir('results')
        np.savetxt('compressed_lengths_cts', np.array(compress_lengths))

        state = rans.unflatten(compressed_message)
        decode_start_time = time.time()
        reconstructed_data_points = []

        print("\nDecoding data points ...")
        for n in tqdm(range(len(data_points))):
            state, data_point_ = self.vae_pop(state)
            reconstructed_data_points.insert(0, data_point_.numpy()[0])

        print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

        recovered_bits = rans.flatten(state)
        assert all(other_bits == recovered_bits)
        np.testing.assert_equal(reconstructed_data_points, self.data_points_singles)
        print('\nLossless reconstruction!')