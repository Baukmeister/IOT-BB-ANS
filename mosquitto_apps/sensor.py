import io

import paho.mqtt.client as mqtt
import torch
from tqdm import tqdm

from util.HouseholdPowerDataLoader import HouseholdPowerDataset

client = mqtt.Client()
client.connect("localhost", 1883, 60)

householdPowerDataSet = HouseholdPowerDataset(
    "../data/household_power_consumption",
    caching=False
)

buffer = io.BytesIO()

for idx in tqdm(range(householdPowerDataSet.__len__())):
    item = householdPowerDataSet.__getitem__(idx)
    nums = item.numpy()
    for num in nums:
        client.publish("test", int(num))