import os
import time

import numpy as np
import torch
from tqdm import tqdm

import rans
from compression import tvae_utils
from models.beta_binomial_vae import BetaBinomialVAE_sbs
from util import bb_util
from util.SimpleDataLoader import SimpleDataSet

rng = np.random.RandomState(0)

prior_precision = 8
obs_precision = 24
q_precision = 14

data_set_size = 1000

batch_size = 1
pooling_factor = 15
input_dim = 1 * int(pooling_factor)
scale_factor = 100
hidden_dim = 300
latent_dim = 20
val_set_ratio = 0.00
train_batch_size = 64
learning_rate = 0.0001
weight_decay = 0.001
model_type = "beta_binomial_vae"

model_name = "../models/trained_models/simple/simple_model"
dataSet = SimpleDataSet(data_range=scale_factor, pooling_factor=pooling_factor, data_set_size=int(1e6))

compress_lengths = []

latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 3 * int(pooling_factor))
obs_size = np.prod(obs_shape)

## Setup codecs
# VAE codec
model = BetaBinomialVAE_sbs(
    n_features=input_dim,
    range=scale_factor,
    batch_size=train_batch_size,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    lr=learning_rate,
    wc=weight_decay,
    plot=False
)

model.load_state_dict(torch.load(model_name))

model.eval()

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

## Load simple data
data_set = SimpleDataSet(pooling_factor=pooling_factor, data_set_size=int(1e6))
data_points_singles = [data_set.__getitem__(i).cpu().numpy() for i in range(data_set_size)]
num_batches = len(data_points_singles) // batch_size

obs_append = tvae_utils.beta_binomial_obs_append(scale_factor, obs_precision)
obs_pop = tvae_utils.beta_binomial_obs_pop(scale_factor, obs_precision)

vae_append = bb_util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                                prior_precision, q_precision)
vae_pop = bb_util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                          prior_precision, q_precision)

# randomly generate some 'other' bits
other_bits = rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
state = rans.unflatten(other_bits)
data_points = np.split(np.reshape(data_points_singles, (len(data_points_singles), -1)), num_batches)

print_interval = 100
encode_start_time = time.time()
for i, data_point in enumerate(data_points):
    state = vae_append(state, data_point)

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
    state, data_point_ = vae_pop(state)
    reconstructed_data_points.insert(0, data_point_.numpy()[0])

print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

recovered_bits = rans.flatten(state)
assert all(other_bits == recovered_bits)
np.testing.assert_equal(reconstructed_data_points, data_points_singles)
print('\nLossless reconstruction!')
