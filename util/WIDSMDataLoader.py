import os
import pandas as pd
from tqdm import tqdm


class WISDMDataLoader:

    def __init__(self, path) -> None:
        self.path = path
        self.columns = ['user', 'activity', 'time', 'x', 'y', 'z']

        self.phone_accel_path = f"{self.path}/phone/accel"
        self.phone_gyro_path = f"{self.path}/phone/gyro"

        self.watch_accel_path = f"{self.path}/watch/accel"
        self.watch_gyro_path = f"{self.path}/watch/gyro"

        self.data_dict = {
            self.phone_accel_path: pd.DataFrame(columns=self.columns),
            self.phone_gyro_path: pd.DataFrame(columns=self.columns),
            self.watch_accel_path: pd.DataFrame(columns=self.columns),
            self.watch_gyro_path: pd.DataFrame(columns=self.columns)
        }

    def load(self):

        for path in self.data_dict.keys():

            print(f"\nLoading data from '{path}' ...")
            for dirname, _, filenames in os.walk(path):
                for filename in tqdm(filenames):
                    if not str(filename).startswith("."):
                        df = pd.read_csv(
                            f"{path}/{filename}",
                            sep=",", header=None)
                        temp = pd.DataFrame(data=df.values, columns=self.columns)
                        self.data_dict[path] = pd.concat([self.data_dict[path], temp])
