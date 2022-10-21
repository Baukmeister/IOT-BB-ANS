import os
import time

import numpy as np
import torch
from autograd.builtins import tuple as ag_tuple

import craystack as cs
from compression.torch_util import torch_fun_to_numpy_fun
from craystack import bb_ans
from models.vae import VAE_full
from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name

rng = np.random.RandomState(0)

prior_precision = 16
q_precision = 16

batch_size = 1
data_set_size = 10000
pooling_factor = 1
hidden_dim = 32
latent_dim = 2
discretize = True
obs_precision = 16
compress_lengths = []

latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 3 * int(pooling_factor))
obs_size = np.prod(obs_shape)

## Setup codecs
# VAE codec
model = VAE_full(n_features=3 * int(pooling_factor), batch_size=batch_size, hidden_size=hidden_dim, latent_size=latent_dim,
                 device="cpu")
model.load_state_dict(torch.load(vae_model_name(
    model_folder="../models",
    dicretize=discretize,
    hidden_dim=hidden_dim,
    latent_dim=latent_dim,
    pooling_factor=pooling_factor
)))

model.eval()

encoder_net = torch_fun_to_numpy_fun(model.encoder)
decoder_net = torch_fun_to_numpy_fun(model.decoder)


# obs_codec is used to generate the likelihood function P(X|Z).
# mean and stdd are for a distribution over the output X variable based on a specific z value!
def obs_codec(res):
    return cs.DiagGaussian_UnifBins(mean=res[0], stdd=res[1], bin_min=60, bin_max=100, n_bins=150, coding_prec=obs_precision)


def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                     np.reshape(head[latent_size:], obs_shape)))


## Load biometrics data
data_set = WISDMDataset("../data/wisdm-dataset/raw", pooling_factor=pooling_factor, discretize=discretize)
data_points_singles = [data_set.__getitem__(i).cpu().numpy() for i in range(data_set_size)]
num_batches = len(data_points_singles) // batch_size

vae_append, vae_pop = cs.repeat(cs.substack(
    bb_ans.VAE(decoder_net, encoder_net, obs_codec, prior_precision, q_precision),
    vae_view), num_batches)

data_points = np.split(np.reshape(data_points_singles, (len(data_points_singles), -1)), num_batches)
data_points = np.int64(data_points)



## Encode
# Initialize message with some 'extra' bits
encode_t0 = time.time()
init_message = cs.base_message(obs_size + latent_size)

# Encode the datapoints
message, = vae_append(init_message, data_points)

flat_message = cs.flatten(message)
encode_t = time.time() - encode_t0

print("All encoded in {:.2f}s.".format(encode_t))

message_len = 32 * len(flat_message)
print("Used {} bits.".format(message_len))
print("This is {:.4f} bits per data_point.".format(message_len / len(data_points)))

## Decode
decode_t0 = time.time()
message = cs.unflatten(flat_message, obs_size + latent_size)

message, data_points_ = vae_pop(message)
decode_t = time.time() - decode_t0

print('All decoded in {:.2f}s.'.format(decode_t))

np.testing.assert_equal(data_points, data_points_)
