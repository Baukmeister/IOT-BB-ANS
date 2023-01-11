import paho.mqtt.client as mqtt
import sys


class LeaderNode():

    def __init__(self, data_set_type):

        self.client = None
        self.data_set_type = data_set_type
        self.set_up_connection()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("test")

    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))

    def set_up_connection(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.connect("localhost", 1883, 60)
        self.client.loop_forever()


if __name__ == "__main__":

    data_set_type = sys.argv[1]
    LeaderNode(data_set_type)