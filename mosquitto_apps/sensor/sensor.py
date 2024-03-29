import json
import sys
import time

import paho.mqtt.client
import paho.mqtt.client as mqtt
from tqdm import tqdm

from compression import benchmark_compression
from util.DataLoadersLite.HouseholdPowerDataLoader_Lite import HouseholdPowerDataset_Lite
from util.DataLoadersLite.IntelLabDataLoader_Lite import IntelLabDataset_Lite
from util.DataLoadersLite.SimpleDataLoader_Lite import SimpleDataSet_Lite
from util.DataLoadersLite.WIDSMDataLoader_Lite import WISDMDataset_Lite
from util.experiment_params import Params

# TODO document that -1 input for sensor_idx means using only one sensor for all data
# TODO document that -1 input for compression_samples_num means using the entire dataset
class SensorNode:

    def __init__(self, host_address, sensor_idx, param_path, compression_mode="neural"):
        self.data_set = None
        self.client: paho.mqtt.client.Client = None

        with open(f"{param_path}", ) as f:
            params_json = json.load(f)
            self.params: Params = Params.from_dict(params_json)

        self.data_set_name = self.params.data_set_name
        self.data_set_dir = self.params.test_data_set_dir
        self.compression_mode = compression_mode
        self.host_address = host_address
        if sensor_idx == -1:
            self.sensor_idx = "total"
        else:
            self.sensor_idx = sensor_idx

        self.mosquitto_port = 1883
        self.load_data()

        if compression_mode == "neural":
            self.set_up_connection()
            self.client.loop_start()
            self.send_data()
            self.client.loop_stop()

        elif compression_mode == "benchmark":
            self.send_data()

    def load_data(self):

        if self.data_set_name == "simple":
            self.data_set = SimpleDataSet_Lite(
                data_range=self.params.scale_factor,
                data_set_size=self.params.compression_samples_num,
                pooling_factor=self.params.pooling_factor
            )
        elif self.data_set_name == "household":
            self.data_set = HouseholdPowerDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.params.scale_factor,
                sensor_idx=self.sensor_idx
            )
        elif self.data_set_name == "wisdm":
            self.data_set = WISDMDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.params.scale_factor,
                sensor_idx=self.sensor_idx
            )
        elif self.data_set_name == "intel":
            self.data_set = IntelLabDataset_Lite(
                self.data_set_dir,
                scaling_factor=self.params.scale_factor,
                sensor_idx=self.sensor_idx
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

        if self.params.compression_samples_num == -1:
            compression_sample_num = self.data_set.__len__()
            print(f"MAX sample number: {compression_sample_num}")
            compression_samples = [self.data_set.__getitem__(idx) for idx in range(compression_sample_num)]
        else:
            compression_samples = [self.data_set.__getitem__(idx) for idx in range(self.params.compression_samples_num)]

        if self.compression_mode == "neural":
            total_sent_messages = 0

            self.client.publish(self.data_set_name, "SOT", qos=2)
            total_sent_messages += 1

            for nums in tqdm(compression_samples):
                json_list = json.dumps(list(nums))
                self.client.publish(self.data_set_name, json_list, qos=2)
                total_sent_messages += 1
                time.sleep(0.01)

            # end of transmission
            self.client.publish(self.data_set_name, "EOT", qos=2)
            total_sent_messages += 1

            print(f"Sent a total of {total_sent_messages} messages (Including start and stop messages)")
        elif self.compression_mode == "benchmark":

            print(f"Benchmarking compression on dataset '{self.params.data_set_name}' for sensor '{self.sensor_idx}'")
            benchmark_compression.benchmark_on_data(compression_samples)


if __name__ == "__main__":
    param_path = sys.argv[1]
    sensor_idx = int(sys.argv[2])
    compression_mode = sys.argv[3]
    if len(sys.argv) >= 5:
        host_address = sys.argv[4]
    else:
        host_address = "localhost"

    SensorNode(host_address, sensor_idx, param_path, compression_mode)
