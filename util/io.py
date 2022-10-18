def vae_model_name(dicretize, hidden_dim, latent_dim, pooling_factor):
    model_name = f"../models/trained_vae_pooling{pooling_factor}_l{latent_dim}_h{hidden_dim}_d{dicretize}"
    return model_name
