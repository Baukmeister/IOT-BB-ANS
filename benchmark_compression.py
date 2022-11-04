import gzip
import bz2
import lzma
import numpy as np
import io

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


if __name__ == "__main__":
    # Biometrics data
    data_set_size = 600
    dataSet = WISDMDataset("data/wisdm-dataset/raw", pooling_factor=15, discretize=True, scaling_factor=10000)
    data = np.array([dataSet.__getitem__(i).cpu().numpy() for i in range(data_set_size)]).astype("uint8")
    bench_compressor(gzip_compress, gzip.decompress, "gzip", data)
    bench_compressor(bz2_compress, bz2.decompress, "bz2", data)
    bench_compressor(lzma_compress, lzma.decompress, "lzma", data)
