import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset
from tqdm import tqdm


class WISDMDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        if not self.caching:
            item = torch.tensor(self.WISDMdf.iloc[index:index + self.pooling_factor, 2:5].values.flatten()).float()
        else:
            item = self.cached_data_samples[index]

        return item.to(self.device)

    def __len__(self) -> int:
        if self.caching:
            return len(self.cached_data_samples)
        else:
            return self.WISDMdf.shape[0] // self.pooling_factor


    def __init__(self, path, pooling_factor=1, discretize=False, scaling_factor=1000, shift=False,
                 data_set_size="single", caching=True) -> None:

        self.pooling_factor = pooling_factor
        self.discretize = discretize
        self.scaling_factor = scaling_factor
        self.path = path
        self.caching = caching
        self.pkl_name = f"pf_{pooling_factor}disc_{discretize}scale_{scaling_factor}shift_{shift}ds_{data_set_size}.pkl"
        self.pkl_path = f"{self.path}/{self.pkl_name}"
        self.data_set_size = data_set_size
        self.columns = ['user', 'time', 'x', 'y', 'z']

        self.phone_accel_path = f"{self.path}/phone/accel"
        self.phone_gyro_path = f"{self.path}/phone/gyro"

        self.watch_accel_path = f"{self.path}/watch/accel"
        self.watch_gyro_path = f"{self.path}/watch/gyro"

        self.WISDMdf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self.cached_data_samples = []

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

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
            print("\nLoading cached data samples ...")
            with open(self.pkl_path, 'rb') as f:
                self.cached_data_samples = pickle.load(f)
        else:
            self._load()
            if self.caching:
                self._set_up_cache()
                with open(self.pkl_path, 'wb') as f:
                    pickle.dump(self.cached_data_samples, f)

        if shift:
            mins = abs(self.WISDMdf.min())

            self.WISDMdf['x'] += mins['x']
            self.WISDMdf['y'] += mins['y']
            self.WISDMdf['z'] += mins['z']

    def _load(self):

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
                        temp['time'] = temp['time'].astype(np.float)

                        if self.discretize:
                            temp['x'] = ((temp['x'].astype(np.float) * self.scaling_factor)).round()
                            temp['y'] = ((temp['y'].astype(np.float) * self.scaling_factor)).round()
                            temp['z'] = ((temp['z'].astype(np.float) * self.scaling_factor)).round()
                        else:
                            temp['x'] = temp['x'].astype(np.float)
                            temp['y'] = temp['y'].astype(np.float)
                            temp['z'] = temp['z'].astype(np.float)

                        self.WISDMdf = pd.concat([self.WISDMdf, temp])

    def _set_up_cache(self):
        print("\nCaching data samples ...")
        for idx in tqdm(range(self.__len__())):
            item = torch.tensor(self.WISDMdf.iloc[idx:idx + self.pooling_factor, 2:5].values.flatten()).float()
            self.cached_data_samples.append(item)
