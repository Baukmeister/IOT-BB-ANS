import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset
from tqdm import tqdm


class WISDMDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        if not self.caching:
            item = self.WISDMdf.iloc[index:index + self.pooling_factor, 2:5].values.flatten()
        else:
            with open(self._cached_file_name(index), "rb") as f:
                item = np.load(f)

        return torch.tensor(item).float()

    def __len__(self) -> int:
        if self.caching:
            return len(self.cached_file_list)
        else:
            return self.WISDMdf.shape[0] // self.pooling_factor

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
        self.columns = ['user', 'time', 'x', 'y', 'z']
        self.shift = shift

        self.phone_accel_path = f"{self.path}/phone/accel"
        self.phone_gyro_path = f"{self.path}/phone/gyro"

        self.watch_accel_path = f"{self.path}/watch/accel"
        self.watch_gyro_path = f"{self.path}/watch/gyro"

        self.WISDMdf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self.cached_data_samples = []

        if self.data_set_size == "single":
            self.paths = [self.phone_accel_path]
        elif self.data_set_size == "all":
            self.paths = [self.phone_accel_path, self.phone_gyro_path, self.watch_accel_path, self.watch_gyro_path]
        elif self.data_set_size == "gyro":
            self.paths = [self.phone_gyro_path, self.watch_gyro_path]
        elif self.data_set_size == "accel":
            self.paths = [self.phone_accel_path, self.watch_accel_path]
        else:
            raise ValueError(f"Invalid size option: {self.data_set_size}")

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
            for dirname, _, filenames in os.walk(path):
                for filename in tqdm(filenames):
                    if not str(filename).startswith("."):
                        df = pd.read_csv(
                            f"{path}/{filename}",
                            sep=",", header=None)[[0, 2, 3, 4, 5]]
                        temp = pd.DataFrame(data=df.values, columns=self.columns)

                        temp['z'] = temp['z'].str.replace(';', '')
                        temp['user'] = temp['user'].astype(np.double)
                        temp['time'] = temp['time'].astype(float)

                        if self.discretize:
                            temp['x'] = ((temp['x'].astype(float) * self.scaling_factor)).round()
                            temp['y'] = ((temp['y'].astype(float) * self.scaling_factor)).round()
                            temp['z'] = ((temp['z'].astype(float) * self.scaling_factor)).round()
                        else:
                            temp['x'] = temp['x'].astype(float)
                            temp['y'] = temp['y'].astype(float)
                            temp['z'] = temp['z'].astype(float)

                        self.WISDMdf = pd.concat([self.WISDMdf, temp])

        if self.shift:
            shift_vals = abs(self.WISDMdf.min())

            self.WISDMdf['x'] += shift_vals['x']
            self.WISDMdf['y'] += shift_vals['y']
            self.WISDMdf['z'] += shift_vals['z']

        if self.caching:
            print("Storing samples in cache...")
            for idx in tqdm(range(self.WISDMdf.shape[0] // self.pooling_factor)):
                item = self.WISDMdf.iloc[idx:idx + self.pooling_factor, 2:5].values.flatten()

                with open(self._cached_file_name(idx), "wb") as f:
                    np.save(f, item)

        self.range = self.WISDMdf.iloc[:, 2:5].max().max() - self.WISDMdf.iloc[:, 2:5].min().min()

    def _cached_file_name(self, idx):
        return f"{self.pkl_path}/{idx}.npy"
