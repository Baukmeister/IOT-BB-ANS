import os
import json
import sys

import paho.mqtt.client
import paho.mqtt.client as mqtt
from tqdm import tqdm

from util.DataLoadersLite.HouseholdPowerDataLoader_Lite import HouseholdPowerDataset_Lite
from util.DataLoadersLite.IntelLabDataLoader_Lite import IntelLabDataset_Lite
from util.DataLoadersLite.SimpleDataLoader_Lite import SimpleDataSet_Lite
from util.DataLoadersLite.WIDSMDataLoader_Lite import WISDMDataset_Lite
from util.experiment_params import Params


class SensorNode():

    def __init__(self, data_set_dir, host_address, model_param_path):
        self.data_set = None
        self.client: paho.mqtt.client.Client = None

        with open(f"{model_param_path}") as f:
            params_json = json.load(f)
            self.params: Params = Params.from_dict(params_json)

        self.host_address = host_address
        self.data_set_name = self.params.data_set_name
        self.data_set_dir = data_set_dir
        self.mosquitto_port = 1883

        self.load_data()
        self.set_up_connection()
        self.client.loop_start()
        self.send_data()
        self.client.loop_stop()

    def load_data(self):

        if self.data_set_name == "simple":
            self.data_set = SimpleDataSet_Lite(
                data_range=self.params.scale_factor,
                pooling_factor=self.params.pooling_factor
            )
        elif self.data_set_name == "household":
            self.data_set = HouseholdPowerDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.params.scale_factor,
                caching=False,
            )
        elif self.data_set_name == "wisdm":
            self.data_set = WISDMDataset_Lite(
                self.data_set_dir,
                pooling_factor=self.params.pooling_factor,
                discretize=self.params.discretize,
                scaling_factor=self.params.scale_factor,
                shift=self.params.shift,
                data_set_size=self.params.data_set_type,
                caching=False
            )
        elif self.data_set_name == "intel":
            self.data_set = IntelLabDataset_Lite(
                self.data_set_dir,
                pooling_factor=self.params.pooling_factor,
                scaling_factor=self.params.scale_factor,
                caching=self.params.caching,
                metric=self.params.metric
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
        total_sent_messages = 1

        # TODO: change these hardcoded values
        # for idx in tqdm(range(self.params.compression_samples_num)):
        for idx in tqdm(range(self.params.pooling_factor * self.params.compression_samples_num)):
            item = self.data_set.__getitem__(4000 + idx)
            nums = item
            for num in nums:
                self.client.publish(self.data_set_name, int(num), qos=2)
                total_sent_messages += 1

        # end of transmission
        self.client.publish(self.data_set_name, "EOT", qos=2)
        print(f"Sent a total of {total_sent_messages} messages")


if __name__ == "__main__":
    data_set_dir = sys.argv[1]
    model_param_path = sys.argv[2]
    if len(sys.argv) >= 4:
        host_address = sys.argv[3]
    else:
        host_address = "localhost"

    SensorNode(data_set_dir, host_address, model_param_path)
