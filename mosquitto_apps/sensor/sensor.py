import io
import sys

import paho.mqtt.client as mqtt
from tqdm import tqdm

from util.DataLoadersLite.HouseholdPowerDataLoader_Lite import HouseholdPowerDataset_Lite


class SensorNode():

    def __init__(self, data_set_type, data_set_dir, host_address):
        self.data_set = None
        self.client = None
        self.host_address = host_address
        self.data_set_type = data_set_type
        self.data_set_dir = data_set_dir

        self.load_data()
        self.set_up_connection()
        self.send_data()

    def load_data(self):

        if self.data_set_type == "household":
            self.data_set = HouseholdPowerDataset_Lite(
                self.data_set_dir,
                scaling_factor=100,
                caching=False
            )
        else:
            print(f"Data set type {self.data_set_type} not implemented!")

    def set_up_connection(self):
        self.client = mqtt.Client()
        self.client.connect(self.host_address, 1883, 60)

    def send_data(self):
        for idx in tqdm(range(self.data_set.__len__() // 10)):
            item = self.data_set.__getitem__(idx)
            nums = item
            for num in nums:
                self.client.publish("test", int(num))


if __name__ == "__main__":
    data_set_type = sys.argv[1]
    data_set_dir = sys.argv[2]
    if len(sys.argv) >= 4:
        host_address = sys.argv[3]
    else:
        host_address = "localhost"

    SensorNode(data_set_type, data_set_dir, host_address)
