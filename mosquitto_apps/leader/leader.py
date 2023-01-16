import json
import sys

import paho.mqtt.client as mqtt

from compression.Neural_Compressor import NeuralCompressor
from util.io import input_dim
from util.experiment_params import Params


class LeaderNode():

    def __init__(self, data_set_type, host_address, model_param_path):

        self.compressor: NeuralCompressor = None
        self.model_name = None
        self.n_features = None

        with open(model_param_path) as f:
            params_json = json.load(f)
            self.model_params: Params = Params.from_json(params_json)

        self.client = None
        self.compression_batch_size = 10
        self.buffer = []
        self.host_address = host_address
        self.data_set_type = data_set_type
        self.instantiate_neural_compressor()
        self.set_up_connection()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("test")

    def on_message(self, client, userdata, msg):

        self.buffer.append(msg.payload)

        if len(self.buffer) == self.compression_batch_size:
            self.compress_current_buffer()

    def compress_current_buffer(self):
        self.compressor.add_to_state(self.buffer)

    def set_up_connection(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(host_address, 1883, 60)
        self.client.loop_forever()

    def instantiate_neural_compressor(self):

        n_features = input_dim(self.model_params)
        self.compressor = NeuralCompressor(params=self.model_params, data_samples=[], input_dim=n_features)


if __name__ == "__main__":

    data_set_type = sys.argv[1]
    model_param_path = sys.argv[2]
    if len(sys.argv) >= 4:
        host_address = sys.argv[3]
    else:
        host_address = "localhost"

    LeaderNode(data_set_type, host_address, model_param_path)
