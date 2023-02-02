from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Params:
    n_feature: int = 3
    data_set_name: str = "simple"
    pooling_factor: int = 15
    hidden_dim: int = 300
    latent_dim: int = 20
    train_set_ratio: float = 1.0
    val_set_ratio: float = 0.2
    test_set_ratio: float = 0.2
    train_batch_size: int = 64
    discretize: bool = True
    learning_rate: float = 1e-8
    weight_decay: float = 0.01
    scale_factor: int = 100
    range: int = 100
    shift: bool = False
    model_type: str = "beta_binomial_vae"
    metric = "all"
    train_batches: int = 10000
    max_epochs: int = 1
    compression_batch_size: int = 1
    prior_precision: int = 8
    obs_precision: int = 24
    q_precision: int = 14
    compression_samples_num: int = 1000
    data_set_type: str = ""
    caching: bool = False
    use_first_samples_as_extra_bits: bool = True
    random_bit_samples: int = 50
    test_data_set_dir: str = "../data/test_data_dfs"
