import json
import sys
import time

import paho.mqtt.client
import paho.mqtt.client as mqtt
from tqdm import tqdm

from util.DataLoadersLite.HouseholdPowerDataLoader_Lite import HouseholdPowerDataset_Lite
from util.DataLoadersLite.IntelLabDataLoader_Lite import IntelLabDataset_Lite
from util.DataLoadersLite.SimpleDataLoader_Lite import SimpleDataSet_Lite
from util.DataLoadersLite.WIDSMDataLoader_Lite import WISDMDataset_Lite
from util.experiment_params import Params


class SensorNode:

    def __init__(self, host_address, sensor_idx, param_path):
        self.data_set = None
        self.client: paho.mqtt.client.Client = None

        with open(f"{param_path}",) as f:
            params_json = json.load(f)
            self.params: Params = Params.from_dict(params_json)

        self.host_address = host_address
        self.data_set_name = self.params.data_set_name
        self.data_set_dir = self.params.test_data_set_dir
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
                sensor_idx=sensor_idx
            )
        elif self.data_set_name == "wisdm":
            self.data_set = WISDMDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.params.scale_factor,
                sensor_idx=sensor_idx
            )
        elif self.data_set_name == "intel":
            self.data_set = IntelLabDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.params.scale_factor,
                sensor_idx=sensor_idx
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
        total_sent_messages = 0

        self.client.publish(self.data_set_name, "SOT", qos=2)
        total_sent_messages += 1

        for idx in tqdm(range(self.params.compression_samples_num)):
            nums = self.data_set.__getitem__(idx)
            json_list = json.dumps(list(nums))
            self.client.publish(self.data_set_name, json_list, qos=2)
            total_sent_messages += 1
            time.sleep(0.01)

        # end of transmission
        self.client.publish(self.data_set_name, "EOT", qos=2)
        total_sent_messages += 1

        print(f"Sent a total of {total_sent_messages} messages (Including start and stop messages)")


if __name__ == "__main__":
    param_path = sys.argv[1]
    sensor_idx = sys.argv[2]
    if len(sys.argv) >= 4:
        host_address = sys.argv[3]
    else:
        host_address = "localhost"

    SensorNode(host_address, sensor_idx, param_path)
