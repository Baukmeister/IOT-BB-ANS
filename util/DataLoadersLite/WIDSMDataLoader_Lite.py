import pickle

import pandas as pd


class WISDMDataset_Lite:

    def __getitem__(self, index):

        item = self.WISDMdf.iloc[index:index + self.pooling_factor, 2:5].values.flatten()

        return item

    def __len__(self) -> int:

        return self.WISDMdf.shape[0] // self.pooling_factor

    def __init__(self, path, pooling_factor=1, scaling_factor=1,  metric="all") -> None:


        self.pooling_factor = pooling_factor
        self.scaling_factor = scaling_factor
        self.path = path
        self.pkl_name = f"wisdm.pkl"
        self.pkl_path = f"{self.path}/{self.pkl_name}"
        self.columns = ['user', 'time', 'x', 'y', 'z']
        self.metric = metric

        self.phone_accel_path = f"{self.path}/phone/accel"
        self.phone_gyro_path = f"{self.path}/phone/gyro"

        self.watch_accel_path = f"{self.path}/watch/accel"
        self.watch_gyro_path = f"{self.path}/watch/gyro"

        self.WISDMdf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self.cached_data_samples = []

        if self.metric == "single":
            self.paths = [self.phone_accel_path]
        elif self.metric == "all":
            self.paths = [self.phone_accel_path, self.phone_gyro_path, self.watch_accel_path, self.watch_gyro_path]
        elif self.metric == "gyro":
            self.paths = [self.phone_gyro_path, self.watch_gyro_path]
        elif self.metric == "accel":
            self.paths = [self.phone_accel_path, self.watch_accel_path]
        else:
            raise ValueError(f"Invalid metric option: {self.metric}")

        self._load()

    def _load(self):

        with open(self.pkl_path, 'rb') as f:
            self.WISDMdf = pickle.load(f)

        self.range = self.WISDMdf.iloc[:, 2:5].max().max() - self.WISDMdf.iloc[:, 2:5].min().min()
