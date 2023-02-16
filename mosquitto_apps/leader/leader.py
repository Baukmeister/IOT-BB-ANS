import json
import sys
import threading
from queue import Queue

import numpy as np
import paho.mqtt.client
import paho.mqtt.client as mqtt

from compression.Neural_Compressor import NeuralCompressor
from util.io import input_dim
from util.experiment_params import Params

class LeaderNode:

    def __init__(self, host_address, model_param_path):

        self.sensor_finished_counter = 0
        self.sensor_started_counter = 0
        self.compressor: NeuralCompressor = None
        self.model_name = None
        self.n_features = None
        self.mosquitto_port = 1883
        self.data_points_num = 0
        self.compression_steps = 0

        with open(f"{model_param_path}") as f:
            params_json = json.load(f)
            self.params: Params = Params.from_dict(params_json)

        self.input_dim = input_dim(self.params)
        self.client: paho.mqtt.client.Client = None
        self.random_bits_filled = not self.params.use_first_samples_as_extra_bits
        self.random_bits_size = 50
        self.random_bits_buffer = []

        self.compression_batch_size = self.input_dim
        self.buffer = []
        self.host_address = host_address
        self.data_set_name = self.params.data_set_name
        self.sample_queue = Queue()
        self.exit = False

        self.instantiate_neural_compressor()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe(topic=self.data_set_name, qos=2)

    def on_message(self, client, userdata, msg: paho.mqtt.client.MQTTMessage):

        raw_payload = msg.payload
        self.sample_queue.put(raw_payload)

    def handle_next_sample(self, raw_payload):
        if raw_payload == b"SOT":
            self.sensor_started_counter += 1
        elif raw_payload == b"EOT":
            self.sensor_finished_counter += 1

            if self.sensor_finished_counter == self.sensor_started_counter:
                print("All sensors finished!")
                if self.random_bits_filled:
                    self.finish_input_processing()
                else:
                    raise RuntimeWarning(
                        "Random bits have not been filled! Consider increasing the number of samples sent by sensor"
                    )

        else:
            payload = json.loads(raw_payload)

            self.process_message_neural_compressor(payload)

    def process_message_neural_compressor(self, payload: list):
        if self.random_bits_filled:
            for item in payload:
                self.buffer.append(int(item))
            if len(self.buffer) == self.compression_batch_size:
                self.compress_current_buffer()
                self.buffer = []
        else:
            self.random_bits_buffer.append(payload)
            buffer_len = sum([len(l) for l in self.random_bits_buffer])
            if buffer_len >= self.random_bits_size:
                print(f"Using first {buffer_len} samples as random bits for ANS coder")

                random_bits_samples = [int(sample) for sublist in self.random_bits_buffer for sample in sublist]
                self.compressor.set_random_bits(np.array(random_bits_samples))
                self.random_bits_filled = True

    def handle_data_buffer(self):
        while not self.exit:
            next_sample = self.sample_queue.get()
            self.handle_next_sample(next_sample)

    def compress_current_buffer(self):
        data_point = np.array(self.buffer)
        data_point.shape = (1, self.input_dim)
        self.compression_steps += 1
        self.compressor.add_to_state(data_point)
        print(f"Compressed sample {self.compression_steps * self.params.pooling_factor}"
              f"/{self.params.compression_samples_num * self.sensor_started_counter} ")
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

        if self.params.use_first_samples_as_extra_bits:
            include_init_bits_in_stats = True
        else:
            include_init_bits_in_stats = False

        self.compressor.decode_entire_state(self.compression_steps)

        print(f"Printing metrics for {self.data_points_num} data points")
        self.compressor.get_encoding_stats(self.data_points_num,
                                           include_init_bits_in_calculation=include_init_bits_in_stats)
        self.compressor.plot_stack_sizes()

        self.exit = True


if __name__ == "__main__":

    model_param_path = sys.argv[1]
    if len(sys.argv) >= 3:
        host_address = sys.argv[2]
    else:
        host_address = "localhost"

    leader = LeaderNode(host_address, model_param_path)

    mqtt_thread = threading.Thread(target=leader.set_up_connection)
    ans_coder_thread = threading.Thread(target=leader.handle_data_buffer)

    mqtt_thread.start()
    ans_coder_thread.start()

    sys.stdout.flush()
    mqtt_thread.join()
    ans_coder_thread.join()
