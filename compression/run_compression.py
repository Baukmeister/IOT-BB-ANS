import os
import time

import numpy as np
import torch
from autograd.builtins import tuple as ag_tuple

from compression.torch_util import torch_fun_to_numpy_fun
from craystack import bb_ans
from models.beta_binomial_vae import BetaBinomialVAE_sbs
from models.vae import VAE_full
from models.vanilla_vae import Vanilla_VAE
from util.WIDSMDataLoader import WISDMDataset
from util.io import vae_model_name

rng = np.random.RandomState(0)

prior_precision = 16
q_precision = 16

batch_size = 1
data_set_size = 100
obs_precision = 24
compress_lengths = []

# MODEL CONFIG
pooling_factor = 1
input_dim = 3 * int(pooling_factor)
hidden_dim = 200
latent_dim = 25
val_set_ratio = 0.00
train_batch_size = 32
dicretize = True
learning_rate = 0.001
weight_decay = 0.0001
scale_factor = 10
shift = True
model_type = "beta_binomial_vae"
data_set_type = "accel"

latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 3 * int(pooling_factor))
obs_size = np.prod(obs_shape)

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
    range=160 * scale_factor,
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

encoder_net = torch_fun_to_numpy_fun(model.encoder)
decoder_net = torch_fun_to_numpy_fun(model.decoder)


# obs_codec is used to generate the likelihood function P(X|Z).
# mean and stdd are for a distribution over the output X variable based on a specific z value!
def obs_codec(res):
    # return cs.DiagGaussian_StdBins(mean=res[0], stdd=res[1], coding_prec=obs_precision, bin_prec=20)
    # return cs.DiagGaussian_GaussianBins(mean=res[0], stdd=res[1],bin_mean=res[0], bin_stdd=res[1], coding_prec=obs_precision, bin_prec=16)
    return cs.Logistic_UnifBins()
    return cs.DiagGaussian_UnifBins(mean=res[0], stdd=res[1], bin_min=0, bin_max=160 * scale_factor, coding_prec=obs_precision, n_bins=160 * scale_factor)


def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                     np.reshape(head[latent_size:], obs_shape)))


## Load biometrics data
data_set = WISDMDataset("../data/wisdm-dataset/raw", pooling_factor=pooling_factor, discretize=dicretize,
                        scaling_factor=scale_factor, shift=shift)
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
print("This is {:.4f} bits per data_point.".format(message_len / np.size(data_points)))

## Decode
decode_t0 = time.time()
message = cs.unflatten(flat_message, obs_size + latent_size)

message, data_points_ = vae_pop(message)
decode_t = time.time() - decode_t0

print('All decoded in {:.2f}s.'.format(decode_t))

np.testing.assert_equal(data_points, data_points_)
