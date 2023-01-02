from compression.Neural_Compressor import NeuralCompressor
from experiment_pipelines.experiment_params import Params
from model_training.Simple_train_VAE import SimpleVaeTrainer
from util.SimpleDataLoader import SimpleDataSet
from torch.utils import data


def main():

    experiments_to_run = ["simple", "household"]


    if "simple" in experiments_to_run:

        print("Running model training for simple data ...")

        simple_data_params = Params(
            input_dim=1,
            train_set_ratio=0.2,
            val_set_ratio=0.01
        )

        simpleDataSet = SimpleDataSet(
            data_range=simple_data_params.scale_factor,
            pooling_factor=simple_data_params.pooling_factor,
            data_set_size=int(1e6)
        )

        testSetSize = int(len(simpleDataSet) * simple_data_params.test_set_ratio)
        trainSetSize = len(simpleDataSet) - (testSetSize)
        train_set, test_set = data.random_split(simpleDataSet, [trainSetSize, testSetSize])

        # only provide the "real" train_set to the VAE trainer
        simple_vae_trainer = SimpleVaeTrainer(simple_data_params, train_set)
        simple_vae_trainer.train_model()

        simple_neural_compressor = NeuralCompressor(simple_data_params,test_set, "simple")
        simple_neural_compressor.run_compression()


        #Todo: Add benchmark compression on test_set

if __name__ == "__main__":
    main()