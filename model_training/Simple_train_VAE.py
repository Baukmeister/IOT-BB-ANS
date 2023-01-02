import torch
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils.data import Dataset

from experiment_pipelines.experiment_params import Params
from model_training.VAE_Trainer import VaeTrainer


class SimpleVaeTrainer(VaeTrainer):

    def __init__(self, params: Params, dataSet: Dataset):
        input_dim = int(1 * params.pooling_factor)
        super().__init__(params, dataSet, "Simple", input_dim)


