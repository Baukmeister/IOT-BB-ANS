import os
import json
import sys

import paho.mqtt.client
import paho.mqtt.client as mqtt
from tqdm import tqdm

from util.DataLoadersLite.HouseholdPowerDataLoader_Lite import HouseholdPowerDataset_Lite
from util.experiment_params import Params


class SensorNode():

    def __init__(self, data_set_dir, host_address, model_param_path):
        self.data_set = None
        self.client: paho.mqtt.client.Client = None

        with open(f"{model_param_path}") as f:
            params_json = json.load(f)
            self.model_params: Params = Params.from_dict(params_json)

        self.host_address = host_address
        self.data_set_name  = self.model_params.data_set_name
        self.data_set_dir = data_set_dir
        self.mosquitto_port = 1883



        self.load_data()
        self.set_up_connection()
        self.send_data()

    def load_data(self):

        if self.data_set_name == "household":
            self.data_set = HouseholdPowerDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.model_params.scale_factor,
                caching=False
            )
        else:
            print(f"Data set name {self.data_set_name} not implemented!")

    def set_up_connection(self):
        self.client = mqtt.Client()
        try:
            self.client.connect(host_address, self.mosquitto_port)
        except ConnectionRefusedError as e:
            raise Warning(f"Could not connect at port {self.mosquitto_port} - Make sure the service is running!")

    def send_data(self):
        for idx in tqdm(range(self.data_set.__len__() // 100000)):
            item = self.data_set.__getitem__(idx)
            nums = item
            for num in nums:
                self.client.publish(self.data_set_name, int(num), )

        # end of transmission
        self.client.publish(self.data_set_name, "EOT")


if __name__ == "__main__":
    data_set_dir = sys.argv[1]
    model_param_path = sys.argv[2]
    if len(sys.argv) >= 4:
        host_address = sys.argv[3]
    else:
        host_address = "localhost"

    SensorNode(data_set_dir, host_address, model_param_path)
