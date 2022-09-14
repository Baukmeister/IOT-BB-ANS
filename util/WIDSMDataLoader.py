import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset
from tqdm import tqdm


class WISDMDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        return torch.tensor(self.userDfs[index].values)

    def __len__(self) -> int:
        return len(self.userDfs)

    def __init__(self, path) -> None:
        self.path = path
        self.columns = ['user', 'time', 'x', 'y', 'z']

        self.phone_accel_path = f"{self.path}/phone/accel"
        self.phone_gyro_path = f"{self.path}/phone/gyro"

        self.watch_accel_path = f"{self.path}/watch/accel"
        self.watch_gyro_path = f"{self.path}/watch/gyro"

        self.WISDMdf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self._load()

    def _load(self):

        for path in [self.phone_accel_path]:  # , self.phone_gyro_path, self.watch_accel_path, self.watch_gyro_path]:

            print(f"\nLoading data from '{path}' ...")
            for dirname, _, filenames in os.walk(path):
                for filename in tqdm(filenames):
                    if not str(filename).startswith("."):
                        df = pd.read_csv(
                            f"{path}/{filename}",
                            sep=",", header=None)[[0,2,3,4,5]]
                        temp = pd.DataFrame(data=df.values, columns=self.columns)

                        temp['z'] = temp['z'].str.replace(';', '')
                        temp['user'] = temp['user'].astype(np.float)
                        temp['time'] = temp['time'].astype(np.float)
                        temp['x'] = temp['x'].astype(np.float)
                        temp['y'] = temp['y'].astype(np.float)
                        temp['z'] = temp['z'].astype(np.float)

                        self.WISDMdf = pd.concat([self.WISDMdf, temp])

        print("Partitioning into user datasets ...")
        for user in tqdm(self.WISDMdf.user.unique()):
            self.userDfs.append(self.WISDMdf[self.WISDMdf.user == user])
