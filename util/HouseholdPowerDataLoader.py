import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import T_co, Dataset
from tqdm import tqdm


class HouseholdPowerDataset(Dataset):

    def __getitem__(self, index) -> T_co:
        if not self.caching:
            item = self.HouseholdPowerDf.iloc[index:index + self.pooling_factor, self.item_indices].values.flatten()

        else:
            with open(self._cached_file_name(index), "rb") as f:
                item = np.load(f)

        return torch.tensor(item).float()

    def __len__(self) -> int:
        if self.caching:
            return len(self.cached_file_list)
        else:
            return self.HouseholdPowerDf.shape[0] // self.pooling_factor

    def __init__(self, path, pooling_factor=1, discretize=True, scaling_factor=1, caching=True, metric="all") -> None:

        self.path = path
        self.pooling_factor = pooling_factor
        self.discretize = discretize
        self.scaling_factor = scaling_factor
        self.caching = caching
        self.metric = metric
        self.pkl_name = f"household_power_data"
        self.pkl_path = f"{self.path}/cache/{self.pkl_name}"
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

        self.HouseholdPowerDf = pd.DataFrame(columns=self.columns)
        self.userDfs = []
        self.cached_data_samples = []

        # TODO: Adapt
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



        self.HouseholdPowerDf = pd.read_csv(f"{self.path}/household_power_consumption.txt", sep=";", low_memory=False)


        #remove '?' rows
        self.HouseholdPowerDf = self.HouseholdPowerDf.where(self.HouseholdPowerDf['Global_active_power'] != '?')

        self.HouseholdPowerDf['Global_active_power'].fillna(0, inplace=True)
        self.HouseholdPowerDf['Global_reactive_power'].fillna(0, inplace=True)
        self.HouseholdPowerDf['Voltage'].fillna(0, inplace=True)
        self.HouseholdPowerDf['Global_intensity'].fillna(0, inplace=True)
        self.HouseholdPowerDf['Sub_metering_1'].fillna(0, inplace=True)
        self.HouseholdPowerDf['Sub_metering_2'].fillna(0, inplace=True)
        self.HouseholdPowerDf['Sub_metering_3'].fillna(0, inplace=True)


        if self.discretize:
            self.HouseholdPowerDf['Global_active_power'] = (
                        self.HouseholdPowerDf['Global_active_power'].astype(np.float) * self.scaling_factor).round()
            self.HouseholdPowerDf['Global_reactive_power'] = (
                        self.HouseholdPowerDf['Global_reactive_power'].astype(np.float) * self.scaling_factor).round()
            self.HouseholdPowerDf['Voltage'] = (
                        self.HouseholdPowerDf['Voltage'].astype(np.float) * self.scaling_factor).round()
            self.HouseholdPowerDf['Global_intensity'] = (
                        self.HouseholdPowerDf['Global_intensity'].astype(np.float) * self.scaling_factor).round()
            self.HouseholdPowerDf['Sub_metering_1'] = (
                        self.HouseholdPowerDf['Sub_metering_1'].astype(np.float) * self.scaling_factor).round()
            self.HouseholdPowerDf['Sub_metering_2'] = (
                        self.HouseholdPowerDf['Sub_metering_2'].astype(np.float) * self.scaling_factor).round()
            self.HouseholdPowerDf['Sub_metering_3'] = (
                        self.HouseholdPowerDf['Sub_metering_3'].astype(np.float) * self.scaling_factor).round()

        if self.caching:
            print("Storing samples in cache...")
            for idx in tqdm(range(self.HouseholdPowerDf.shape[0] // self.pooling_factor)):
                item = self.HouseholdPowerDf.iloc[idx:idx + self.pooling_factor, 2:9].values.flatten()

                with open(self._cached_file_name(idx), "wb") as f:
                    np.save(f, item)

        self.range = self.HouseholdPowerDf.iloc[:, self.item_indices].max().max() - self.HouseholdPowerDf.iloc[:,
                                                                                    self.item_indices].min().min()

    def _cached_file_name(self, idx):
        return f"{self.pkl_path}/{idx}.npy"
