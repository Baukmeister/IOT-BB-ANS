from benchmark_compression import benchmark_on_data
from compression.Neural_Compressor import NeuralCompressor
from experiment_pipelines.experiment_params import Params
from model_training.VAE_Trainer import VaeTrainer
from util.HouseholdPowerDataLoader import HouseholdPowerDataset
from util.SimpleDataLoader import SimpleDataSet
from torch.utils import data


def main():
    experiments_to_run = [
        # "simple",
        "household"
    ]

    if "simple" in experiments_to_run:
        print("_" * 25)
        print("Running model training for simple data ...")

        simple_params = Params(
            train_set_ratio=0.2,
            val_set_ratio=0.01
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
        simple_vae_trainer = VaeTrainer(simple_params, train_set, "simple", simple_input_dim)
        simple_vae_trainer.train_model()

        print("Running neural compression for simple data ...")

        simple_neural_compressor = NeuralCompressor(simple_params, test_set, "simple", simple_input_dim)
        simple_neural_compressor.run_compression()

        print("Running benchmark compression for simple data ...")
        benchmark_on_data(test_set)
        print("_" * 25)

    if "household" in experiments_to_run:
        print("_" * 25)
        print("Running model training for household_power data ...")

        household_power_params = Params(
            train_set_ratio=0.1,
            val_set_ratio=0.005,
            scale_factor=100,
            pooling_factor=5,
            hidden_dim=50,
            latent_dim=10,
            train_batch_size=8,
            discretize=True,
            learning_rate=0.0001,
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

        # only provide the "real" train_set to the VAE trainer
        if household_power_params.metric == "all":
            householder_power_input_dim = 7 * int(household_power_params.pooling_factor)
        else:
            householder_power_input_dim = 1 * int(household_power_params.pooling_factor)

        simple_vae_trainer = VaeTrainer(household_power_params, train_set, "HouseholdPower",
                                        householder_power_input_dim)
        simple_vae_trainer.train_model()

        print("Running neural compression for household_power data ...")

        household_power_neural_compressor = NeuralCompressor(household_power_params, test_set, "HouseholdPower",
                                                             householder_power_input_dim)
        household_power_neural_compressor.run_compression()

        print("Running benchmark compression for household_power data ...")
        benchmark_on_data(test_set)
        print("_" * 25)

    elif "WISDM" in experiments_to_run:
        # TODO
        pass
    elif "intel_lab" in experiments_to_run:
        # TODO
        pass


if __name__ == "__main__":
    main()
