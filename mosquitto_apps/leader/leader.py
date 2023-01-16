import json
import sys

import numpy
import numpy as np
import paho.mqtt.client
import paho.mqtt.client as mqtt

from compression.Neural_Compressor import NeuralCompressor
from util.io import input_dim
from util.experiment_params import Params

# TODO: Implement idea of using first few samples as random bits
# TODO: Add benchmark compressor leader variant
class LeaderNode():

    def __init__(self,host_address, model_param_path):

        self.compressor: NeuralCompressor = None
        self.model_name = None
        self.n_features = None
        self.mosquitto_port = 1883
        self.data_points_num = 0
        self.compression_steps = 0

        with open(f"{model_param_path}") as f:
            params_json = json.load(f)
            self.model_params: Params = Params.from_dict(params_json)

        self.input_dim = input_dim(self.model_params)
        self.client: paho.mqtt.client.Client = None
        self.compression_batch_size = self.input_dim
        self.buffer = []
        self.host_address = host_address
        self.data_set_name = self.model_params.data_set_name
        self.instantiate_neural_compressor()
        self.set_up_connection()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe(topic=self.data_set_name, qos=2)

    def on_message(self, client, userdata, msg: paho.mqtt.client.MQTTMessage):

        payload = msg.payload
        #print(payload)

        if payload == b"EOT":
            print("Finished compression!")
            self.end_compression()
        else:
            self.buffer.append(int(msg.payload))

        if len(self.buffer) == self.compression_batch_size:
            self.compress_current_buffer()
            self.buffer = []

    def compress_current_buffer(self):
        data_point = np.array(self.buffer)
        data_point.shape = (1, self.input_dim)
        print(f"Encoding current buffer! ({self.compression_steps})")
        self.compression_steps += 1
        self.compressor.add_to_state(data_point)
        self.data_points_num += data_point.size


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
        self.compressor = NeuralCompressor(params=self.model_params, data_samples=[], input_dim=n_features, plot=True)

    def end_compression(self):
        self.client.unsubscribe(self.model_params.data_set_name)
        self.client.loop_stop()
        self.client.disconnect()
        self.compressor.get_encoding_stats(self.data_points_num)
        self.compressor.plot_stack_sizes()

if __name__ == "__main__":

    model_param_path = sys.argv[1]
    if len(sys.argv) >= 3:
        host_address = sys.argv[2]
    else:
        host_address = "localhost"

    LeaderNode(host_address, model_param_path)
