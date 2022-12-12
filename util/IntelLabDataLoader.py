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
            item = None #TODO
        else:
            with open(self._cached_file_name(index), "rb") as f:
                item = np.load(f)

        return torch.tensor(item).float()

    def __len__(self) -> int:
        if self.caching:
            return len(self.cached_file_list)
        else:
            return self.IntelDataDf.shape[0] // self.pooling_factor

    def __init__(self, path, pooling_factor=1, discretize=False, scaling_factor=1000, shift=False,
                 data_set_size="single", caching=True) -> None:

        self.pooling_factor = pooling_factor
        self.discretize = discretize
        self.scaling_factor = scaling_factor
        self.path = path
        self.caching = caching
        self.pkl_name = f"pf_{pooling_factor}-disc_{discretize}-scale_{scaling_factor}-shift_{shift}-ds_{data_set_size}"
        self.pkl_path = f"{self.path}/cache/{self.pkl_name}"
        self.data_set_size = data_set_size
        self.columns = [] #TODO
        self.shift = shift


        self.IntelDataDf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self.cached_data_samples = []

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

        for path in self.paths:

            print(f"\nLoading data from '{path}' ...")

            #TODO: Load data.txt file

            self.IntelDataDf = pd.concat([self.IntelDataDf, temp])


        if self.caching:
            print("Storing samples in cache...")
            for idx in tqdm(range(self.IntelDataDf.shape[0] // self.pooling_factor)):
                item = self.IntelDataDf.iloc[idx:idx + self.pooling_factor, 2:5].values.flatten()

                with open(self._cached_file_name(idx), "wb") as f:
                    np.save(f, item)

    def _cached_file_name(self, idx):
        return f"{self.pkl_path}/{idx}.npy"
