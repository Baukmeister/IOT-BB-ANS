import gzip
import bz2
import lzma
import time

import numpy as np
import zstd
from tqdm import tqdm


def bench_compressor(compress_fun, decompress_fun, compressor_name, data_points):
    print("#########################")
    print(f"Number of samples: {np.size(data_points)}")
    start = time.time()
    byts = compress_fun(data_points)
    end = time.time()
    compression_time = end - start
    n_bits = len(byts) * 8
    uncompressed_bits = (np.size(data_points)*32)
    compression_rate = n_bits/uncompressed_bits
    bits_per_datapoint = (n_bits / np.size(data_points))
    print(f"Compressor: {compressor_name}. \n"
          f"Uncompressed data size: {_format_bytes(uncompressed_bits)}\n"
          f"Compression rate: {round(compression_rate,4)}. \n"
          f"BpD: {bits_per_datapoint}. \n"
          f"Time: {compression_time} seconds")
    print("#########################\n")
    recon = np.ndarray(data_points.shape, np.uint8, decompress_fun(byts))
    assert np.equal(data_points, recon).all()


def gzip_compress(data_points):
    original_size = np.size(data_points)
    return gzip.compress(data_points.tobytes())


def bz2_compress(data_points):
    original_size = np.size(data_points)
    return bz2.compress(data_points.tobytes())


def lzma_compress(data_points):
    original_size = np.size(data_points)
    return lzma.compress(data_points.tobytes())

def zstd_compress(data_points):
    original_size = np.size(data_points)
    return zstd.compress(data_points.tobytes())


def benchmark_on_data(custom_data_set, compression_samples_num=None):

    if compression_samples_num is None:
        compression_samples_num = len(custom_data_set)

    if type(custom_data_set) == list:
        custom_data = np.array(custom_data_set).astype("uint8")
    else:
        custom_data = np.hstack([custom_data_set.__getitem__(i).cpu().numpy() for i in tqdm(range(compression_samples_num))]).astype("uint8")

    bench_compressor(gzip_compress, gzip.decompress, "gzip", custom_data)
    bench_compressor(bz2_compress, bz2.decompress, "bz2", custom_data)
    bench_compressor(lzma_compress, lzma.decompress, "lzma", custom_data)
    bench_compressor(zstd_compress, zstd.decompress, "zstd", custom_data)


def _format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size} {power_labels[n]}"
