import torch
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils import data
from torch.utils.data import Dataset

from experiment_pipelines.experiment_params import Params
from model_training.VAE_Trainer import VaeTrainer
import pytorch_lightning as pl


class SimpleVaeTrainer(VaeTrainer):

    def __init__(self, params: Params, dataSet: Dataset):
        input_dim = int(1 * params.pooling_factor)
        super().__init__(params, "Simple", input_dim)
        self.params = params
        self.dataSet = dataSet

    def train_model(self):
        # CONFIG
        valSetSize = int(len(self.dataSet) * self.val_set_ratio)
        trainSetSize = len(self.dataSet) - valSetSize
        train_set, val_set = data.random_split(self.dataSet, [trainSetSize, valSetSize])

        trainDataLoader = data.DataLoader(train_set, batch_size=self.params.train_batch_size, shuffle=True, num_workers=1)
        valDataLoader = data.DataLoader(val_set)

        profiler = SimpleProfiler()
        # profiler = PyTorchProfiler()

        trainer = pl.Trainer(limit_train_batches=self.params.train_batches, max_epochs=1, accelerator='gpu', devices=1, profiler=profiler)
        trainer.fit(model=self.model, train_dataloaders=trainDataLoader, val_dataloaders=valDataLoader)
        torch.save(self.model.state_dict(), self.model_name)
