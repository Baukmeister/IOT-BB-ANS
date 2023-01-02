import gzip
import bz2
import lzma
import numpy as np
import io

from tqdm import tqdm

from util.HouseholdPowerDataLoader import HouseholdPowerDataset
from util.IntelLabDataLoader import IntelLabDataset
from util.SimpleDataLoader import SimpleDataSet
from util.WIDSMDataLoader import WISDMDataset


def bench_compressor(compress_fun, decompress_fun, compressor_name, data_points):
    print(f"Running compressor: {compressor_name}")
    byts = compress_fun(data_points)
    n_bits = len(byts) * 8
    bits_per_datapoint = (n_bits / np.size(data_points))
    print(f"Compressor: {compressor_name}. Rate: {bits_per_datapoint} bits per data point.")
    recon = np.ndarray(data_points.shape, np.uint8, decompress_fun(byts))
    assert np.equal(data_points, recon).all()


def gzip_compress(data_points):
    original_size = np.size(data_points)
    assert data_points.dtype is np.dtype('uint8')
    return gzip.compress(data_points.tobytes())


def bz2_compress(data_points):
    original_size = np.size(data_points)
    assert data_points.dtype is np.dtype('uint8')
    return bz2.compress(data_points.tobytes())


def lzma_compress(data_points):
    original_size = np.size(data_points)
    assert data_points.dtype is np.dtype('uint8')
    return lzma.compress(data_points.tobytes())


def benchmark_on_data(custom_data_set):
    custom_data = np.array([custom_data_set.__getitem__(i).cpu().numpy()[0] for i in tqdm(range(len(custom_data_set)))]).astype(
        "uint8")
    bench_compressor(gzip_compress, gzip.decompress, "gzip", custom_data)
    bench_compressor(bz2_compress, bz2.decompress, "bz2", custom_data)
    bench_compressor(lzma_compress, lzma.decompress, "lzma", custom_data)


if __name__ == "__main__":
    # Biometrics data
    data_set_size = 1000
    data_set_name = "intelLab"

    if data_set_name == "WISDM":
        dataSet = WISDMDataset(
            "data/wisdm-dataset/raw",
            pooling_factor=100,
            discretize=True,
            scaling_factor=100,
            caching=False,
            data_set_size="all"
        )

    elif data_set_name == "simple":
        dataSet = SimpleDataSet()

    elif data_set_name == "intelLab":
        dataSet = IntelLabDataset(
            path="data/IntelLabData",
            pooling_factor=10,
            scaling_factor=100,
            caching=False,
            metric="temperature"
        )

    elif data_set_name == "householdPower":
        dataSet = HouseholdPowerDataset(
            "data/household_power_consumption",
            pooling_factor=5,
            scaling_factor=100,
            caching=False,
            metric="all")

    print("\nCollecting data for compression benchmark ...")
    data = np.array([dataSet.__getitem__(i).cpu().numpy()[0] for i in tqdm(range(data_set_size))]).astype("uint8")
    bench_compressor(gzip_compress, gzip.decompress, "gzip", data)
    bench_compressor(bz2_compress, bz2.decompress, "bz2", data)
    bench_compressor(lzma_compress, lzma.decompress, "lzma", data)
