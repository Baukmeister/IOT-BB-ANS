def vae_model_name(model_folder, dicretize, hidden_dim, latent_dim, pooling_factor):
    model_name = f"{model_folder}/trained_vae_pooling{pooling_factor}_l{latent_dim}_h{hidden_dim}_d{dicretize}"
    return model_name
