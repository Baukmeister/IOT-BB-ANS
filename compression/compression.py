# TODO work through these tutorials to gain more insights https://github.com/facebookresearch/NeuralCompression


import sys
import torch
import numpy as np
from autograd.builtins import tuple as ag_tuple
import craystack as cs
from models.vae import VAE_full
from torch_vae import BinaryVAE
from torch_util import torch_fun_to_numpy_fun
import craystack.bb_ans as bb_ans
import time

from util.WIDSMDataLoader import WISDMDataset

rng = np.random.RandomState(0)

prior_precision = 8
bernoulli_precision = 16
q_precision = 14

batch_size = 1
data_set_size = 500
latent_dim = 9
latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 3)
obs_size = np.prod(obs_shape)

## Setup codecs
# VAE codec
model = VAE_full(n_features=3, hidden_size=18, latent_size=latent_dim, device="cpu")
model.load_state_dict(torch.load('../models/trained_vae_l9_h18'))

encoder_net = torch_fun_to_numpy_fun(model.encoder)
decoder_net = torch_fun_to_numpy_fun(model.decoder)

obs_codec = lambda p: cs.Bernoulli(p, bernoulli_precision)


def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                     np.reshape(head[latent_size:], obs_shape)))


## Load biometrics data
data_set = WISDMDataset("../data/wisdm-dataset/raw")
data_points = [entry for entry in data_set.WISDMdf.iloc[:data_set_size,][["x", "y", "z"]].to_numpy()]

num_batches = len(data_points) // batch_size

vae_append, vae_pop = cs.repeat(cs.substack(
    bb_ans.VAE(decoder_net, encoder_net, obs_codec, prior_precision, q_precision),
    vae_view), num_batches)

## Encode
# Initialize message with some 'extra' bits
encode_t0 = time.time()
init_message = cs.base_message(obs_size + latent_size)

# TODO: The datapoints need to be seperated into lists of numpy arrays!
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
