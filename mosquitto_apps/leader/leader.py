import json
import sys

import numpy as np
import paho.mqtt.client
import paho.mqtt.client as mqtt

import benchmark_compression
from compression.Neural_Compressor import NeuralCompressor
from util.io import input_dim
from util.experiment_params import Params


class LeaderNode:

    def __init__(self, host_address, model_param_path, compression_mode="neural"):

        self.compression_mode = compression_mode
        self.compressor: NeuralCompressor = None
        self.model_name = None
        self.n_features = None
        self.mosquitto_port = 1883
        self.data_points_num = 0
        self.compression_steps = 0

        with open(f"{model_param_path}") as f:
            params_json = json.load(f)
            self.params: Params = Params.from_dict(params_json)

        print(f"Running leader node in '{self.compression_mode}' compression mode")

        self.input_dim = input_dim(self.params)
        self.client: paho.mqtt.client.Client = None
        self.random_bits_filled = not self.params.use_first_samples_as_extra_bits
        self.random_bits_size = 50
        self.random_bits_buffer = []

        self.compression_batch_size = self.input_dim
        self.buffer = []
        self.host_address = host_address
        self.data_set_name = self.params.data_set_name

        if self.compression_mode == "neural":
            self.instantiate_neural_compressor()

        self.set_up_connection()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe(topic=self.data_set_name, qos=2)

    def on_message(self, client, userdata, msg: paho.mqtt.client.MQTTMessage):

        payload = msg.payload

        if payload == b"EOT":
            print("Finished processing samples!")
            self.finish_input_processing()
        else:
            if self.compression_mode == "neural":
                self.process_message_neural_compressor(int(payload))
            elif self.compression_mode == "benchmark":
                self.buffer.append(payload)
                # Just let the buffer accumulate
                pass

    def process_message_neural_compressor(self, payload: int):
        if self.random_bits_filled:
            self.buffer.append(payload)
            if len(self.buffer) == self.compression_batch_size:
                self.compress_current_buffer()
                self.buffer = []
        else:
            self.random_bits_buffer.append(payload)
            if len(self.random_bits_buffer) == self.random_bits_size:
                print(f"Using first {self.random_bits_size} samples as random bits for ANS coder")
                self.compressor.set_random_bits(np.array(self.random_bits_buffer))
                self.random_bits_filled = True

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
            print(f"Trying to connect to {host_address}:{self.mosquitto_port}")
            self.client.connect(host_address, self.mosquitto_port)
        except ConnectionRefusedError as e:
            raise Warning(f"Could not connect at port {self.mosquitto_port} - Make sure the service is running!")
        self.client.loop_forever()

    def instantiate_neural_compressor(self):

        if self.host_address == "localhost":
            trained_model_folder = "../models/trained_models"
        else:
            trained_model_folder = "./models/trained_models"

        n_features = input_dim(self.params)
        self.compressor = NeuralCompressor(params=self.params, data_samples=[], input_dim=n_features, plot=True,
                                           trained_model_folder=trained_model_folder)

    def finish_input_processing(self):
        self.client.unsubscribe(self.params.data_set_name)
        self.client.loop_stop()
        self.client.disconnect()
        if self.compression_mode == "neural":
            if self.params.use_first_samples_as_extra_bits:
                include_init_bits_in_stats = True
            else:
                include_init_bits_in_stats = False

            self.compressor.get_encoding_stats(self.data_points_num, include_init_bits_in_calculation=include_init_bits_in_stats)
            self.compressor.plot_stack_sizes()
        elif self.compression_mode == "benchmark":
            benchmark_compression.benchmark_on_data(self.buffer)
            pass


if __name__ == "__main__":

    compression_mode = sys.argv[1]
    model_param_path = sys.argv[2]
    if len(sys.argv) >= 4:
        host_address = sys.argv[3]
    else:
        host_address = "localhost"

    LeaderNode(host_address, model_param_path, compression_mode)
