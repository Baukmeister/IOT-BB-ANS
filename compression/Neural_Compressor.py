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


class NeuralCompressor:

    def __init__(self, params: Params, data_samples: list, input_dim, plot=False,
                 trained_model_folder="../models/trained_models"):
        self.compressed_message = None
        self.bits_per_datapoint_consider_init = None
        self.bits_per_datapoint_no_init = None
        self.compression_rate_consider_init = None
        self.compression_rate_no_init = None
        self.compressed_bits_consider_init = None
        self.compressed_bits_no_init = None
        self.plot = plot
        self.params = params
        self.state = None

        self.data_samples = data_samples
        self.name = params.model_type
        self.n_features = input_dim

        self.model_name = vae_model_name(params=params, folder_root=trained_model_folder)

        self.num_batches = len(self.data_samples) // self.params.compression_batch_size

        vae_full = VAE_full(
            n_features=self.n_features,
            range=self.params.range,
            batch_size=self.params.train_batch_size,
            hidden_dim=self.params.hidden_dim,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay,
            plot=False
        )

        vanilla_vae = Vanilla_VAE(
            n_features=self.n_features,
            scale_factor=self.params.range,
            hidden_dims=None,
            latent_dim=self.params.latent_dim,
            lr=self.params.learning_rate,
            wc=self.params.weight_decay
        )

        beta_binomial_vae = BetaBinomialVAE_sbs(
            n_features=self.n_features,
            range=self.params.range,
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
            self.model.load_state_dict(torch.load(self.model_name))
            self.model.eval()
            obs_append = tvae_utils.gaussian_obs_append(self.model.range.item(), self.params.obs_precision)
            obs_pop = tvae_utils.gaussian_obs_pop(self.model.range.item(), self.params.obs_precision)
        elif self.params.model_type == "vanilla_vae":
            print("Using Vanilla VAE")
            self.model = vanilla_vae
            # TODO: add this
            obs_append = None
            obs_pop = None
        elif self.params.model_type == "beta_binomial_vae":
            print("Using Beta Binomial VAE (beta-binomial likelihood)")
            self.model = beta_binomial_vae
            self.model.load_state_dict(torch.load(self.model_name))
            self.model.eval()
            obs_append = tvae_utils.beta_binomial_obs_append(self.model.range.item(), self.params.obs_precision)
            obs_pop = tvae_utils.beta_binomial_obs_pop(self.model.range.item(), self.params.obs_precision)
        else:
            raise ValueError(f"No model defined for '{self.params.model_type}'")

        rec_net = tvae_utils.torch_fun_to_numpy_fun(self.model.encode)
        gen_net = tvae_utils.torch_fun_to_numpy_fun(self.model.decode)

        # set up compression methods
        latent_shape = (self.params.compression_batch_size, self.params.latent_dim)

        self.vae_append = bb_util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                                             self.params.prior_precision, self.params.q_precision)
        self.vae_pop = bb_util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                                       self.params.prior_precision, self.params.q_precision)
        self.rng = numpy.random.RandomState()
        self.other_bit_integers = self.rng.randint(low=1 << 16, high=1 << 31, size=self.params.random_bit_samples,
                                                   dtype=np.uint32)
        self.state = rans.unflatten(self.other_bit_integers)
        self.stack_sizes = []

        self.encoding_times = []
        self.rec_net_times = []
        self.gen_net_times = []
        self.decoding_times = []

    def run_compression(self):

        data_points = self.encode_data_set()
        self.decode_entire_state(len(data_points))

        if self.plot:
            self.plot_stack_sizes()

    def plot_stack_sizes(self):
        # plot_stack size:
        x = list(range(len(self.stack_sizes)))
        plt.plot(x, self.stack_sizes)
        plt.title(f"Stack size per per sample")
        plt.show()

    def decode_entire_state(self, data_points_num):
        decode_start_time = time.time()
        reconstructed_data_points = []
        self.compressed_message = rans.flatten(self.state)
        print("\nDecoding data points ...")
        for _ in tqdm(range(data_points_num)):
            self.remove_from_state(reconstructed_data_points)
        print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))
        recovered_bits = rans.flatten(self.state)
        assert all(self.other_bit_integers == recovered_bits)
        if data_points_num > 0:
            assert (list(self.data_samples)[0] == list(reconstructed_data_points)[0]).all()
            print('\nLossless reconstruction!')
        else:
            print("\nNo data points were decoded!")

    def encode_data_set(self):
        data_points = np.split(np.reshape(self.data_samples, (len(self.data_samples), -1)), self.num_batches)
        encode_start_time = time.time()
        for i, data_point in tqdm(enumerate(data_points), total=self.params.compression_samples_num):
            self.add_to_state(data_point)
        print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
        compressed_message = self.get_encoding_stats(np.size(data_points))
        self.state = rans.unflatten(compressed_message)
        return data_points

    def get_encoding_stats(self, data_points_num, include_init_bits_in_calculation=False):
        if data_points_num > 0:
            self.compressed_bits_consider_init = 32 * (len(self.compressed_message))
            self.compressed_bits_no_init = 32 * (len(self.compressed_message) - len(self.other_bit_integers))

            self.compression_rate_consider_init = self.compressed_bits_consider_init / (data_points_num * 32)
            self.bits_per_datapoint_consider_init = self.compressed_bits_consider_init / data_points_num

            self.compression_rate_no_init = self.compressed_bits_no_init / (data_points_num * 32)
            self.bits_per_datapoint_no_init = self.compressed_bits_no_init / data_points_num

            self.print_metrics()
        else:
            raise RuntimeWarning("No datapoints encoded on stack! Can't calculate stats!")

    def remove_from_state(self, reconstructed_data_points):
        start = time.time()
        self.state, data_point_ = self.vae_pop(self.state)
        decoding_time = time.time() - start
        self.decoding_times.append(round(decoding_time,3))
        reconstructed_data_points.insert(0, data_point_.numpy()[0].astype(int))

    def add_to_state(self, data_point):
        self.data_samples.append(data_point)
        start = time.time()
        self.state, rec_net_time, gen_net_time = self.vae_append(self.state, data_point)
        encoding_time = time.time() - start
        self.encoding_times.append(round(encoding_time, 5))

        self.rec_net_times.append(round(rec_net_time, 5))
        self.gen_net_times.append(round(gen_net_time, 5))

        current_stack_depth = stack_depth(self.state)

        self.stack_sizes.append(current_stack_depth)

    def set_random_bits(self, random_bits):
        if all(self.other_bit_integers == rans.flatten(self.state)):
            self.state = rans.unflatten(random_bits)
            self.other_bit_integers = random_bits
        else:
            raise ValueError("Can't change random start bits of ANS coder since information has already been encoded!")

    def print_metrics(self):
        print("\n"
              "#########METRICS##########"
              "\n")

        print(f"Used bits (including init bits): {str(self.compressed_bits_consider_init)}")
        print(f"Used bits (without init bits): {str(self.compressed_bits_no_init)}")

        print("\n")

        print(f'Compression ratio (including init bits): {round(self.compression_rate_consider_init, 4)}')
        print(f'Compression ratio (without init bits): {round(self.compression_rate_no_init, 4)}')

        print("\n")

        print(f'BpD (including init bits): {self.bits_per_datapoint_consider_init}')
        print(f'BpD (without init bits): {self.bits_per_datapoint_no_init}')

        print("\n")

        print(f'Stack sizes: {self.stack_sizes}')
        print(f'Encoding times: {self.encoding_times}')
        print(f'Decoding times: {self.decoding_times}')
        print(f'Gen Net times: {self.gen_net_times}')
        print(f'Rec Net times: {self.rec_net_times}')
