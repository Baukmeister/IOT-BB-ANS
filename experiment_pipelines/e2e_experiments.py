from benchmark_compression import benchmark_on_data
from compression.Neural_Compressor import NeuralCompressor
from experiment_pipelines.experiment_params import Params
from model_training.VAE_Trainer import VaeTrainer
from util.HouseholdPowerDataLoader import HouseholdPowerDataset
from util.IntelLabDataLoader import IntelLabDataset
from util.WIDSMDataLoader import WISDMDataset
from util.SimpleDataLoader import SimpleDataSet
from torch.utils import data


def main():
    experiments_to_run = [
        "simple",
        "household",
        #"wisdm",
        #"intel"
    ]

    modes_to_evaluate = [
        #"model_training",
        "compression"
    ]

    if "simple" in experiments_to_run:
        print("_" * 25)

        simple_params = Params(
            train_set_ratio=1.0,
            train_batches=1000,
            val_set_ratio=0.001,
            scale_factor=10000,
            max_epochs=8,
            compression_samples_num=100
        )

        simpleDataSet = SimpleDataSet(
            data_range=simple_params.scale_factor,
            pooling_factor=simple_params.pooling_factor,
            data_set_size=int(1e6)
        )

        testSetSize = int(len(simpleDataSet) * simple_params.test_set_ratio)
        trainSetSize = len(simpleDataSet) - (testSetSize)
        train_set, test_set = data.random_split(simpleDataSet, [trainSetSize, testSetSize])

        # only provide the "real" train_set to the VAE trainer
        simple_input_dim = int(1 * simple_params.pooling_factor)

        if "model_training" in modes_to_evaluate:
            print("Running model training for simple data ...")
            simple_vae_trainer = VaeTrainer(simple_params, train_set, "simple", simple_input_dim)
            simple_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:
            print("Running neural compression for simple data ...")

            simple_neural_compressor = NeuralCompressor(simple_params, test_set, "simple", simple_input_dim)
            simple_neural_compressor.run_compression()

            print("Running benchmark compression for simple data ...")
            benchmark_on_data(test_set, simple_params.compression_samples_num)
        print("_" * 25)

    if "household" in experiments_to_run:
        print("_" * 25)

        household_power_params = Params(
            train_set_ratio=0.2,
            model_type="beta_binomial_vae",
            val_set_ratio=0.005,
            compression_samples_num=10,
            scale_factor=1000,
            pooling_factor=5,
            hidden_dim=50,
            latent_dim=30,
            train_batch_size=8,
            discretize=True,
            learning_rate=0.01,
            train_batches=10000,
            max_epochs=3,
            metric="all"

        )

        householdPowerDataSet = HouseholdPowerDataset(
            "../data/household_power_consumption",
            pooling_factor=household_power_params.pooling_factor,
            scaling_factor=household_power_params.scale_factor,
            caching=False,
            metric=household_power_params.metric
        )

        household_power_params.scale_factor = householdPowerDataSet.range

        testSetSize = int(len(householdPowerDataSet) * household_power_params.test_set_ratio)
        trainSetSize = len(householdPowerDataSet) - (testSetSize)
        train_set, test_set = data.random_split(householdPowerDataSet, [trainSetSize, testSetSize])

        if household_power_params.metric == "all":
            householder_power_input_dim = 7 * int(household_power_params.pooling_factor)
        else:
            householder_power_input_dim = 1 * int(household_power_params.pooling_factor)

        if "model_training" in modes_to_evaluate:
            print("Running model training for household_power data ...")

            # only provide the "real" train_set to the VAE trainer
            household_vae_trainer = VaeTrainer(household_power_params, train_set, "HouseholdPower",
                                               householder_power_input_dim)
            household_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:

            print("Running neural compression for household_power data ...")

            household_power_neural_compressor = NeuralCompressor(household_power_params, test_set, "HouseholdPower",
                                                                 householder_power_input_dim)
            household_power_neural_compressor.run_compression()

            print("Running benchmark compression for household_power data ...")
            benchmark_on_data(test_set, household_power_params.compression_samples_num)
        print("_" * 25)

    if "wisdm" in experiments_to_run:
        print("_" * 25)

        WISDM_params = Params(
            pooling_factor=5,
            compression_samples_num=100,
            hidden_dim=200,
            latent_dim=50,
            train_set_ratio=1.0,
            val_set_ratio=0.01,
            train_batch_size=64,
            discretize=True,
            learning_rate=0.001,
            weight_decay=0.0001,
            scale_factor=10000,
            shift=True,
            max_epochs=8,
            model_type="beta_binomial_vae",
            data_set_type="accel"
        )

        wisdm_dataset = WISDMDataset(
            "../data/wisdm-dataset/raw",
            pooling_factor=WISDM_params.pooling_factor,
            discretize=WISDM_params.discretize,
            scaling_factor=WISDM_params.scale_factor,
            shift=WISDM_params.shift,
            data_set_size=WISDM_params.data_set_type,
            caching=False
        )

        WISDM_params.scale_factor = wisdm_dataset.range

        testSetSize = int(len(wisdm_dataset) * WISDM_params.test_set_ratio)
        trainSetSize = len(wisdm_dataset) - (testSetSize)
        train_set, test_set = data.random_split(wisdm_dataset, [trainSetSize, testSetSize])

        wisdm_power_input_dim = 3 * int(WISDM_params.pooling_factor)

        if "model_training" in modes_to_evaluate:
            print("Running model training for WISDM data ...")

            wisdm_vae_trainer = VaeTrainer(WISDM_params, train_set, "WISDM",
                                           wisdm_power_input_dim)
            wisdm_vae_trainer.train_model()

        if "compression" in modes_to_evaluate:

            print("Running neural compression for WISDM data ...")

            wisdm_neural_compressor = NeuralCompressor(WISDM_params, test_set, "WISDM",
                                                       wisdm_power_input_dim)
            wisdm_neural_compressor.run_compression()

            print("Running benchmark compression for WISDM data ...")
            benchmark_on_data(test_set, WISDM_params.compression_samples_num)
        print("_" * 25)

    if "intel" in experiments_to_run:
        print("_" * 25)

        intel_lab_params = Params(
            pooling_factor=10,
            hidden_dim=50,
            latent_dim=10,
            train_set_ratio=0.2,
            val_set_ratio=0.001,
            train_batch_size=8,
            discretize=True,
            learning_rate=0.0001,
            weight_decay=0.01,
            scale_factor=100,
            shift=True,
            model_type="beta_binomial_vae",
            metric="all"
        )

        intel_lab_dataset = IntelLabDataset(
            "../data/IntelLabData",
            pooling_factor=intel_lab_params.pooling_factor,
            scaling_factor=intel_lab_params.scale_factor,
            caching=intel_lab_params.caching,
            metric=intel_lab_params.metric
        )

        intel_lab_params.scale_factor = intel_lab_dataset.range

        testSetSize = int(len(intel_lab_dataset) * intel_lab_params.test_set_ratio)
        trainSetSize = len(intel_lab_dataset) - (testSetSize)
        train_set, test_set = data.random_split(intel_lab_dataset, [trainSetSize, testSetSize])

        if intel_lab_params.metric == "all":
            intel_lab_input_dim = 4 * int(intel_lab_params.pooling_factor)
        else:
            intel_lab_input_dim = 1 * int(intel_lab_params.pooling_factor)

        if "model_training" in modes_to_evaluate:
            print("Running model training for Intel_Lab data ...")

            intel_lab_vae_trainer = VaeTrainer(intel_lab_params, train_set, "IntelLab",
                                               intel_lab_input_dim)
            intel_lab_vae_trainer.train_model()


        if "compression" in modes_to_evaluate:
            print("Running neural compression for intel_lab data ...")

            intel_lab_neural_compressor = NeuralCompressor(intel_lab_params, test_set, "IntelLab",
                                                           intel_lab_input_dim)
            intel_lab_neural_compressor.run_compression()

            print("Running benchmark compression for intel_lab data ...")
            benchmark_on_data(test_set, intel_lab_params.compression_samples_num    )
        print("_" * 25)


if __name__ == "__main__":
    main()
