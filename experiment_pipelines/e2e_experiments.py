from experiment_pipelines.experiment_params import Params
from model_training.Simple_train_VAE import SimpleVaeTrainer
from util.SimpleDataLoader import SimpleDataSet
from torch.utils import data

simple_data_params = Params()


simpleDataSet = SimpleDataSet(data_range=simple_data_params.scale_factor, pooling_factor=simple_data_params.pooling_factor, data_set_size=int(1e8))
valSetSize = int(len(simpleDataSet) * simple_data_params.val_set_ratio)
trainSetSize = len(simpleDataSet) - valSetSize
train_set, val_set = data.random_split(simpleDataSet, [trainSetSize, valSetSize])


simple_vae_trainer = SimpleVaeTrainer(simple_data_params, simpleDataSet)
simple_vae_trainer.train_model()