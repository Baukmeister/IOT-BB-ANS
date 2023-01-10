import io

import paho.mqtt.client as mqtt
from tqdm import tqdm

from util.HouseholdPowerDataLoader import HouseholdPowerDataset

client = mqtt.Client()
client.connect("localhost", 1883, 60)

householdPowerDataSet = HouseholdPowerDataset(
    "../data/household_power_consumption",
    scaling_factor=100,
    caching=False
)

buffer = io.BytesIO()

for idx in tqdm(range(householdPowerDataSet.__len__()//10)):
    item = householdPowerDataSet.__getitem__(idx)
    nums = item.numpy()
    for num in nums:
        client.publish("test", int(num))