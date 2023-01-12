import paho.mqtt.client as mqtt
import sys


class LeaderNode():

    def __init__(self, data_set_type, host_address):

        self.client = None
        self.compression_batch_size = 10
        self.buffer = []
        self.host_address = host_address
        self.data_set_type = data_set_type
        self.set_up_connection()


    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("test")


    def on_message(self, client, userdata, msg):

        self.buffer.append(msg.payload)

        if len(self.buffer) == self.compression_batch_size:
            self.compress_current_buffer()


    def compress_current_buffer(self):
        #Perform compression
        pass

    def set_up_connection(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect(host_address, 1883, 60)
        self.client.loop_forever()


if __name__ == "__main__":

    data_set_type = sys.argv[1]
    if len(sys.argv) >= 3:
        host_address = sys.argv[2]
    else:
        host_address = "localhost"

    LeaderNode(data_set_type, host_address)