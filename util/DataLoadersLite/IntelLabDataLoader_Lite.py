import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class IntelLabDataset_Lite():

    def __getitem__(self, index):

        item = self.IntelDataDf.iloc[index:index + self.pooling_factor, self.item_indices].values.flatten()

        return item

    def __len__(self) -> int:

        return self.IntelDataDf.shape[0] // self.pooling_factor

    def __init__(self, path, pooling_factor=1, scaling_factor=1,  metric="all") -> None:


        self.path = path
        self.pooling_factor = pooling_factor
        self.scaling_factor = scaling_factor
        self.metric = metric
        self.pkl_name = f"intel_lab_data.pkl"
        self.pkl_path = f"{self.path}/{self.pkl_name}"
        self.columns = ['date', 'time', 'epoch', 'mote_id', 'temperature', 'humidity', 'light', 'voltage']

        self.IntelDataDf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self.cached_data_samples = []

        if self.metric == "all":
            self.item_indices = [4,5,6,7]
        elif self.metric == "temperature":
            self.item_indices = 4
        elif self.metric == "humidity":
            self.item_indices = 5
        elif self.metric == "light":
            self.item_indices = 6
        elif self.metric == "voltage":
            self.item_indices = 7

        self._load()

    def _load(self):

        with open(self.pkl_path, 'rb') as f:
            self.IntelDataDf = pickle.load(f)

        self.range = self.IntelDataDf.iloc[:,self.item_indices].max().max() - self.IntelDataDf.iloc[:,self.item_indices].min().min()


    def _cached_file_name(self, idx):
        return f"{self.pkl_path}/{idx}.npy"
