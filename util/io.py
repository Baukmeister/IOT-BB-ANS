def vae_model_name(model_folder, dicretize, hidden_dim, latent_dim, pooling_factor, scale_factor, model_type, shift):
    model_name = \
        f"{model_folder}/trained_vae_pooling{pooling_factor}_l{latent_dim}_h{hidden_dim}_d{dicretize}_s{scale_factor}_m{model_type}_shift{shift}"
    return model_name
