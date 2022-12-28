from util.SimpleDataLoader import SimpleDataSet
from torch.utils import data

simple_data_params = {
    "pooling_factor": 15,
    "hidden_dim": 300,
    "latent_dim": 20,
    "train_set_ratio": 0.8,
    "val_set_ratio": 0.2,
    "train_batch_size": 64,
    "dicretize": True,
    "learning_rate": 1e-8,
    "weight_decay": 0.01,
    "scale_factor": 100,
    "shift": False,
    "model_type": "beta_binomial_vae",
    "metric": None
}

dataSet = SimpleDataSet(data_range=simple_data_params.scale_factor, pooling_factor=pooling_factor, data_set_size=int(1e8))
valSetSize = int(len(dataSet) * val_set_ratio)
trainSetSize = len(dataSet) - valSetSize
train_set, val_set = data.random_split(dataSet, [trainSetSize, valSetSize])
