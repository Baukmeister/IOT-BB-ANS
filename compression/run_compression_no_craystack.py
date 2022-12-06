import os
import torch
import numpy as np
import util
from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vanilla_vae import Vanilla_VAE
from util import bb_util
import rans
from compression import tvae_utils

from models.vae import VAE_full
import time

from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name

rng = np.random.RandomState(1)

prior_precision = 8
obs_precision = 14
q_precision = 14

data_set_size = 100

# MODEL CONFIG
pooling_factor = 5
input_dim = 3 * int(pooling_factor)
hidden_dim = 500
latent_dim = 10
val_set_ratio = 0.00
train_batch_size = 256
dicretize = True
learning_rate = 0.01
weight_decay = 0.00001
scale_factor = 1
shift = True
model_type = "beta_binomial_vae"
data_set_type = "accel"

compress_lengths = []

latent_shape = (1, latent_dim)
latent_size = np.prod(latent_shape)

## Setup codecs
# VAE codec


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
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    scale_factor=scale_factor,
    batch_size=train_batch_size,
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


model.load_state_dict(torch.load(vae_model_name(
    model_folder="../models/trained_models",
    dicretize=dicretize,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    pooling_factor=pooling_factor,
    scale_factor=scale_factor,
    model_type=model_type,
    shift=shift,
    data_set_type=data_set_type
)))

model.eval()

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

## Load biometrics data
data_set = WISDMDataset("../data/wisdm-dataset/raw", pooling_factor=pooling_factor, discretize=True,
                        scaling_factor=scale_factor, shift=True, caching=False)
data_points_singles = [data_set.__getitem__(i).cpu().numpy() for i in range(data_set_size)]

# obs_append = tvae_utils.gaussian_obs_append(160 * scale_factor, obs_precision)
# obs_pop = tvae_utils.gaussian_obs_pop(160 * scale_factor, obs_precision)
obs_append = tvae_utils.beta_binomial_obs_append(160 * scale_factor, obs_precision)
obs_pop = tvae_utils.beta_binomial_obs_pop(160 * scale_factor, obs_precision)

vae_append = bb_util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                                prior_precision, q_precision)
vae_pop = bb_util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                          prior_precision, q_precision)

# randomly generate some 'other' bits
other_bits = rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
state = rans.unflatten(other_bits)
data_points = np.split(np.reshape(data_points_singles, (len(data_points_singles), -1)), len(data_points_singles))

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

for n in range(len(data_points)):
    state, data_point_ = vae_pop(state)
    original_data_point = data_points[len(data_points) - n - 1]
    reconstructed_data_points.insert(0, data_point_.numpy()[0])

    if not n % print_interval:
        print('Decoded {}'.format(n))

print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

recovered_bits = rans.flatten(state)
assert all(other_bits == recovered_bits)
np.testing.assert_equal(reconstructed_data_points, data_points_singles)
print('\nLossless reconstruction!')
