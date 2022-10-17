import gzip
import bz2
import lzma
import numpy as np

from util.WIDSMDataLoader import WISDMDataset


def bench_compressor(compress_fun, compressor_name, data_points):
    print(f"Running compressor: {compressor_name}")
    byts = compress_fun(data_points)
    n_bits = len(byts) * 8
    bits_per_datapoint = (n_bits / np.size(data_points))
    print(f"Compressor: {compressor_name}. Rate: {bits_per_datapoint} bits per datpoint.")

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


if __name__ == "__main__":
    # Biometrics data
    dataSet = WISDMDataset("data/wisdm-dataset/raw", pooling_factor=1, discretize=True).WISDMdf.to_numpy().astype(np.uint8)
    bench_compressor(gzip_compress, "gzip", dataSet)
    bench_compressor(bz2_compress, "bz2", dataSet)
    bench_compressor(lzma_compress, "lzma", dataSet)


