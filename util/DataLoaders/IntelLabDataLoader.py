import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset
from tqdm import tqdm


class IntelLabDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        if not self.caching:
            item = self.IntelDataDf.iloc[index:index + self.pooling_factor, self.item_indices].values.flatten()

        else:
            with open(self._cached_file_name(index), "rb") as f:
                item = np.load(f)

        return torch.tensor(item).float()

    def __len__(self) -> int:
        if self.caching:
            return len(self.cached_file_list)
        else:
            return self.IntelDataDf.shape[0] // self.pooling_factor

    def __init__(self, path, pooling_factor=1, discretize=True, scaling_factor=1, caching=True, metric="all") -> None:


        self.path = path
        self.pooling_factor = pooling_factor
        self.discretize = discretize
        self.scaling_factor = scaling_factor
        self.caching = caching
        self.metric = metric
        self.pkl_name = f"intel_lab_data"
        self.pkl_path = f"{self.path}/cache/{self.pkl_name}"
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

        if self.caching and os.path.exists(self.pkl_path):
            print("\nSkip data loading in favour of caching ...")
        else:
            self._load(caching=self.caching)

        if self.caching:
            self.cached_file_list = os.listdir(self.pkl_path)

    def _load(self, caching=False):

        if caching:
            if not os.path.exists(self.pkl_path):
                path = Path(self.pkl_path)
                path.mkdir(parents=True)


        self.IntelDataDf = pd.read_csv(f"{self.path}/data.txt", sep=" ", names=self.columns)
        self.IntelDataDf['humidity'].clip(lower=0, upper=100, inplace=True)
        self.IntelDataDf['humidity'].fillna(0, inplace=True)
        self.IntelDataDf['temperature'].clip(lower=0, upper=100, inplace=True)
        self.IntelDataDf['temperature'].fillna(0, inplace=True)
        self.IntelDataDf['light'].fillna(0, inplace=True)

        if self.discretize:
            self.IntelDataDf['humidity'] = (self.IntelDataDf['humidity'].astype(float) * self.scaling_factor).round()
            self.IntelDataDf['temperature'] = (self.IntelDataDf['temperature'].astype(float) * self.scaling_factor).round()
            self.IntelDataDf['light'] = (self.IntelDataDf['light'].astype(float) * self.scaling_factor).round()
            self.IntelDataDf['voltage'] = (self.IntelDataDf['voltage'].astype(float) * self.scaling_factor).round()



        if self.caching:
            print("Storing samples in cache...")
            for idx in tqdm(range(self.IntelDataDf.shape[0] // self.pooling_factor)):
                item = self.IntelDataDf.iloc[idx:idx + self.pooling_factor, 4:8].values.flatten()

                with open(self._cached_file_name(idx), "wb") as f:
                    np.save(f, item)

        self.range = self.IntelDataDf.iloc[:,self.item_indices].max().max() - self.IntelDataDf.iloc[:,self.item_indices].min().min()


    def _cached_file_name(self, idx):
        return f"{self.pkl_path}/{idx}.npy"

    def export_as_csv(self, export_dir):
        file_name = f"{export_dir}/intel_dataset.csv"
        self.IntelDataDf.to_csv(file_name)
