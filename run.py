from util.WIDSMDataLoader import WISDMDataLoader

dataLoader = WISDMDataLoader("data/wisdm-dataset/raw")
dataLoader.load()

print(dataLoader.data_dict)