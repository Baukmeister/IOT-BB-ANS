import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


class HouseholdPowerDataset_Lite:

    def __getitem__(self, index):
        item = self.HouseholdPowerDf.iloc[index:index + self.pooling_factor, self.item_indices].values.flatten()

        return item

    def __len__(self) -> int:
        return self.HouseholdPowerDf.shape[0] // self.pooling_factor

    def __init__(self, path, pooling_factor=1, scaling_factor=1, metric="all", sensor_idx=None) -> None:

        self.path = path
        self.pooling_factor = pooling_factor
        self.scaling_factor = scaling_factor
        self.metric = metric

        if sensor_idx is not None:
            self.pkl_name = f"household_{sensor_idx}.pkl"
        else:
            self.pkl_name = f"household.pkl"


        self.pkl_path = f"{self.path}/{self.pkl_name}"
        self.columns = [
            "Date",
            "Time",
            "Global_active_power",
            "Global_reactive_power",
            "Voltage",
            "Global_intensity",
            "Sub_metering_1",
            "Sub_metering_2",
            "Sub_metering_3"
        ]

        self.HouseholdPowerDf = None
        self.userDfs = []
        self.cached_data_samples = []

        if self.metric == "all":
            self.item_indices = [2, 3, 4, 5, 6, 7, 8]
        elif self.metric == "Global_active_power":
            self.item_indices = 2
        elif self.metric == "Global_reactive_power":
            self.item_indices = 3
        elif self.metric == "Voltage":
            self.item_indices = 4
        elif self.metric == "Global_intensity":
            self.item_indices = 5
        elif self.metric == "Sub_metering_1":
            self.item_indices = 6
        elif self.metric == "Sub_metering_2":
            self.item_indices = 7
        elif self.metric == "Sub_metering_8":
            self.item_indices = 8

        self._load()

    def _load(self):

        try:
            with open(self.pkl_path, 'rb') as f:

                self.HouseholdPowerDf = pickle.load(f)

            self.range = self.HouseholdPowerDf.iloc[:, self.item_indices].max().max() - self.HouseholdPowerDf.iloc[:,
                                                                                        self.item_indices].min().min()
        except FileNotFoundError:
            self.pkl_path = f"../{self.pkl_path}"
            print(f"Data not found in current dir. Trying {self.pkl_path}")
            self._load()
