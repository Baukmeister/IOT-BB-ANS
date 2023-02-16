import json
import pickle
import sys

import numpy
import torch.random
from torch.utils import data

from compression.Neural_Compressor import NeuralCompressor
from compression.benchmark_compression import benchmark_on_data
from model_training.VAE_Trainer import VaeTrainer
from util.DataLoaders.HouseholdPowerDataLoader import HouseholdPowerDataset
from util.DataLoaders.IntelLabDataLoader import IntelLabDataset
from util.DataLoaders.SimpleDataLoader import SimpleDataSet
from util.DataLoaders.WIDSMDataLoader import WISDMDataset
from util.experiment_params import Params
from util.io import input_dim


def main(params_path, test_set_num):

    seed = 11775814
    torch.random.manual_seed(seed)
    numpy.random.seed(seed)

    experiments_to_run = [
        #"simple",
        #"household",
        "wisdm",
        #"intel"
    ]

    modes_to_evaluate = [
        "model_training",
        #"compression"
    ]

    export_to_csv = True

    if "simple" in experiments_to_run:
        print("_" * 25)

        with open(f"{params_path}/simple.json") as f:
            params_json = json.load(f)
            simple_params = Params.from_dict(params_json)


        simpleDataSet = SimpleDataSet(
            data_range=simple_params.scale_factor,
            pooling_factor=simple_params.pooling_factor,
            data_set_size=int(1e6)
        )

        if export_to_csv:
            simpleDataSet.export_as_csv("../data/exports")

        testSetSize = int(len(simpleDataSet) * simple_params.test_set_ratio)
        trainSetSize = len(simpleDataSet) - (testSetSize)
        train_set, test_set = data.random_split(simpleDataSet, [trainSetSize, testSetSize])

        # only provide the "real" train_set to the VAE trainer
        simple_input_dim = input_dim(simple_params)

        if "model_training" in modes_to_evaluate:
            print("Running model training for simple data ...")
            simple_vae_trainer = VaeTrainer(simple_params, train_set, "simple", simple_input_dim)
            simple_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:
            print("Running neural compression for simple data ...")

            test_set_samples = [test_set.__getitem__(i).cpu().numpy() for i in
                                range(simple_params.compression_samples_num)]
            simple_neural_compressor = NeuralCompressor(simple_params, test_set_samples, simple_input_dim)
            simple_neural_compressor.run_compression()

            print("Running benchmark compression for simple data ...")
            benchmark_on_data(test_set, simple_params.compression_samples_num)
        print("_" * 25)

    if "household" in experiments_to_run:
        print("_" * 25)

        with open(f"{params_path}/household.json") as f:
            params_json = json.load(f)
            household_power_params = Params.from_dict(params_json)

        with open("../params/e2e/household.json", "w") as f:
            json.dump(household_power_params.to_dict(), f, indent=4)

        householdPowerDataSet = HouseholdPowerDataset(
            "../data/household_power_consumption",
            pooling_factor=household_power_params.pooling_factor,
            scaling_factor=household_power_params.scale_factor,
            caching=False,
            metric=household_power_params.metric
        )

        if export_to_csv:
            householdPowerDataSet.export_as_csv("../data/exports")

        household_power_params.range = householdPowerDataSet.range

        testSetSize = int(len(householdPowerDataSet) * household_power_params.test_set_ratio)
        trainSetSize = len(householdPowerDataSet) - (testSetSize)
        train_set, test_set = data.random_split(householdPowerDataSet, [trainSetSize, testSetSize])
        _export_to_test_set_dir(household_power_params.test_data_set_dir, test_set.dataset.HouseholdPowerDf,
                                "household", test_set_num)

        householder_power_input_dim = input_dim(household_power_params)

        if "model_training" in modes_to_evaluate:
            print("Running model training for household_power data ...")

            # only provide the "real" train_set to the VAE trainer
            household_vae_trainer = VaeTrainer(household_power_params, train_set, "HouseholdPower",
                                               householder_power_input_dim)
            household_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:
            print("Running neural compression for household_power data ...")

            test_set_samples = [test_set.__getitem__(i).cpu().numpy() for i in
                                range(household_power_params.compression_samples_num)]

            household_power_neural_compressor = NeuralCompressor(household_power_params, test_set_samples,
                                                                 householder_power_input_dim)
            household_power_neural_compressor.run_compression()

            print("Running benchmark compression for household_power data ...")
            benchmark_on_data(test_set, household_power_params.compression_samples_num)
        print("_" * 25)

    if "wisdm" in experiments_to_run:
        print("_" * 25)

        with open(f"{params_path}/wisdm.json") as f:
            params_json = json.load(f)
            wisdm_params = Params.from_dict(params_json)

        with open("../params/e2e/wisdm.json", "w") as f:
            json.dump(wisdm_params.to_dict(), f, indent=4)

        wisdm_dataset = WISDMDataset(
            "../data/wisdm-dataset/raw",
            pooling_factor=wisdm_params.pooling_factor,
            discretize=wisdm_params.discretize,
            scaling_factor=wisdm_params.scale_factor,
            shift=wisdm_params.shift,
            data_set_size=wisdm_params.data_set_type,
            caching=False
        )

        if export_to_csv:
            wisdm_dataset.export_as_csv("../data/exports")

        wisdm_params.range = wisdm_dataset.range

        testSetSize = int(len(wisdm_dataset) * wisdm_params.test_set_ratio)
        trainSetSize = len(wisdm_dataset) - (testSetSize)
        train_set, test_set = data.random_split(wisdm_dataset, [trainSetSize, testSetSize])
        _export_to_test_set_dir(wisdm_params.test_data_set_dir, test_set.dataset.WISDMdf, "wisdm", test_set_num)

        wisdm_power_input_dim = input_dim(wisdm_params)

        if "model_training" in modes_to_evaluate:
            print("Running model training for WISDM data ...")

            wisdm_vae_trainer = VaeTrainer(wisdm_params, train_set, "WISDM",
                                           wisdm_power_input_dim)
            wisdm_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:
            print("Running neural compression for WISDM data ...")

            test_set_samples = [test_set.__getitem__(i).cpu().numpy() for i in
                                range(wisdm_params.compression_samples_num)]
            wisdm_neural_compressor = NeuralCompressor(wisdm_params, test_set_samples,
                                                       wisdm_power_input_dim)
            wisdm_neural_compressor.run_compression()

            print("Running benchmark compression for WISDM data ...")
            benchmark_on_data(test_set, wisdm_params.compression_samples_num)
        print("_" * 25)

    if "intel" in experiments_to_run:
        print("_" * 25)

        with open(f"{params_path}/intel.json") as f:
            params_json = json.load(f)
            intel_lab_params = Params.from_dict(params_json)

        with open("../params/e2e/intel.json", "w") as f:
            json.dump(intel_lab_params.to_dict(), f, indent=4)

        intel_lab_dataset = IntelLabDataset(
            "../data/IntelLabData",
            pooling_factor=intel_lab_params.pooling_factor,
            scaling_factor=intel_lab_params.scale_factor,
            caching=intel_lab_params.caching,
            metric=intel_lab_params.metric
        )

        if export_to_csv:
            intel_lab_dataset.export_as_csv("../data/exports")


        intel_lab_params.range = intel_lab_dataset.range

        testSetSize = int(len(intel_lab_dataset) * intel_lab_params.test_set_ratio)
        trainSetSize = len(intel_lab_dataset) - (testSetSize)
        train_set, test_set = data.random_split(intel_lab_dataset, [trainSetSize, testSetSize])
        _export_to_test_set_dir(intel_lab_params.test_data_set_dir, test_set.dataset.IntelDataDf, "intel", test_set_num)

        intel_lab_input_dim = input_dim(intel_lab_params)

        if "model_training" in modes_to_evaluate:
            print("Running model training for Intel_Lab data ...")

            intel_lab_vae_trainer = VaeTrainer(intel_lab_params, train_set, "IntelLab",
                                               intel_lab_input_dim)
            intel_lab_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:
            print("Running neural compression for intel_lab data ...")

            test_set_samples = [test_set.__getitem__(i).cpu().numpy() for i in
                                range(intel_lab_params.compression_samples_num)]

            intel_lab_neural_compressor = NeuralCompressor(intel_lab_params, test_set_samples,
                                                           intel_lab_input_dim)
            intel_lab_neural_compressor.run_compression()

            print("Running benchmark compression for intel_lab data ...")
            benchmark_on_data(test_set, intel_lab_params.compression_samples_num)
        print("_" * 25)


def _export_to_test_set_dir(dir, df, name, partitions):

    # store entire test dataset

    with open(f"{dir}/{name}_total.pkl", "wb") as f:
        pickle.dump(df, f)

    # store partitions
    partition_dfs = numpy.array_split(df, partitions)

    for partition_idx, partition_df in enumerate(partition_dfs):

        with open(f"{dir}/{name}_{partition_idx}.pkl", "wb") as f:
            pickle.dump(partition_df, f)

if __name__ == "__main__":

    params_path = sys.argv[1]
    if len(sys.argv) >= 3:
        test_set_num = int(sys.argv[2])
    else:
        test_set_num = 3

    main(params_path, test_set_num)
