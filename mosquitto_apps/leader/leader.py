import json
import sys

import numpy
import numpy as np
import paho.mqtt.client as mqtt

from compression.Neural_Compressor import NeuralCompressor
from util.io import input_dim
from util.experiment_params import Params

# TODO: rework this and make it run compression
class LeaderNode():

    def __init__(self,host_address, model_param_path):

        self.compressor: NeuralCompressor = None
        self.model_name = None
        self.n_features = None
        self.mosquitto_port = 1883

        with open(f"{model_param_path}") as f:
            params_json = json.load(f)
            self.model_params: Params = Params.from_dict(params_json)

        self.input_dim = input_dim(self.model_params)
        self.client = None
        self.compression_batch_size = self.input_dim
        self.buffer = []
        self.host_address = host_address
        self.data_set_type = self.model_params.data_set_type
        self.instantiate_neural_compressor()
        self.set_up_connection()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe(self.data_set_type)

    def on_message(self, client, userdata, msg):

        self.buffer.append(int(msg.payload))

        if len(self.buffer) == self.compression_batch_size:
            self.compress_current_buffer()

    def compress_current_buffer(self):
        data_point = np.array(self.buffer)
        data_point.shape = (1, self.input_dim)
        self.compressor.add_to_state(data_point)

    def set_up_connection(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        try:
            self.client.connect(host_address, self.mosquitto_port)
        except ConnectionRefusedError as e:
            raise Warning(f"Could not connect at port {self.mosquitto_port} - Make sure the service is running!")

        self.client.loop_forever()

    def instantiate_neural_compressor(self):

        n_features = input_dim(self.model_params)
        self.compressor = NeuralCompressor(params=self.model_params, data_samples=[], input_dim=n_features)


if __name__ == "__main__":

    model_param_path = sys.argv[1]
    if len(sys.argv) >= 3:
        host_address = sys.argv[2]
    else:
        host_address = "localhost"

    LeaderNode(host_address, model_param_path)
