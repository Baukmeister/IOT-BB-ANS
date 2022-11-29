import os
import torch
import numpy as np
import util
from util import bb_util
import rans
from compression import tvae_utils

from models.vae import VAE_full
import time

from util.SimpleDataLoader import SimpleDataSet
from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name

rng = np.random.RandomState(0)

prior_precision = 8
q_precision = 14

batch_size = 1
data_set_size = 5000
obs_precision = 14
# MODEL CONFIG
pooling_factor = 5
input_dim = 1 * int(pooling_factor)
hidden_dim = 32
latent_dim = 1
val_set_ratio = 0.00
train_batch_size = 16
learning_rate = 0.0001
weight_decay = 0.00001
scale_factor = 1000
shift = True
model_type = "full_vae"
data_set_type = "accel"

model_name = "../models/simple/trained_models/simple_model"
dataSet = SimpleDataSet(pooling_factor=pooling_factor, data_set_size=int(1e6))

compress_lengths = []

latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 3 * int(pooling_factor))
obs_size = np.prod(obs_shape)

## Setup codecs
# VAE codec
model = VAE_full(n_features=input_dim, scale_factor=1, hidden_size=hidden_dim,
                 latent_size=latent_dim,
                 device="cpu")
model.load_state_dict(torch.load(model_name))

model.eval()

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encoder)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decoder)

## Load simple data
data_set = SimpleDataSet(pooling_factor=pooling_factor, data_set_size=int(1e6))
data_points_singles = [data_set.__getitem__(i).cpu().numpy() for i in range(data_set_size)]
num_batches = len(data_points_singles) // batch_size

obs_append = tvae_utils.gaussian_obs_append(100, obs_precision)
obs_pop = tvae_utils.gaussian_obs_pop(100, obs_precision)

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

for n in range(len(data_points)):
    state, data_point_ = vae_pop(state)
    original_data_point = data_points[len(data_points) - n - 1]
    np.testing.assert_allclose(original_data_point, data_point_)

    if not n % print_interval:
        print('Decoded {}'.format(n))

print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

recovered_bits = rans.flatten(state)
assert all(other_bits == recovered_bits)
np.testing.assert_equal(data_point, data_point_)