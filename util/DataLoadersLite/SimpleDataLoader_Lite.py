import numpy as np


class SimpleDataSet_Lite:

    def __getitem__(self, index):
        item = self.data[index:index + self.pooling_factor]
        return item

    def __len__(self) -> int:
        return self.data.size // self.pooling_factor

    def __init__(self, data_range=100, data_set_size=int(1e6), distribution="default", pooling_factor=1) -> None:
        self.pooling_factor = pooling_factor

        if distribution == "default":
            mean = data_range / 2
            std = 1
            #add pooling factor to ensure even split for at least data_set_size samples. Otherwise uneven splits might be created
            data = np.random.normal(mean, scale=std, size=data_set_size + self.pooling_factor)
            self.data =[int(i) for i in data]
