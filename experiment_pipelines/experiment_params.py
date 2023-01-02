from dataclasses import dataclass

@dataclass
class Params:
    input_dim: int
    pooling_factor: int =15
    hidden_dim: int = 300
    latent_dim: int = 20
    train_set_ratio: float = 1.0
    val_set_ratio: float = 0.2
    test_set_ratio: float = 0.1
    train_batch_size: int = 64
    discretize: bool = True
    learning_rate: float = 1e-8
    weight_decay: float = 0.01
    scale_factor: int = 100
    shift: bool = False
    model_type: str = "beta_binomial_vae"
    metric: str = None
    train_batches: int = 10000
    max_epochs = 1
    compression_batch_size = 1
    prior_precision = 8
    obs_precision = 24
    q_precision = 14
    compression_samples_num = 1000

