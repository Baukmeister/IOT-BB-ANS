def vae_model_name(
        model_folder,
        dicretize,
        hidden_dim,
        latent_dim,
        pooling_factor,
        scale_factor,
        model_type,
        shift,
        data_set_type = ""
):
    model_name = f"{model_folder}/trained_vae_pooling{pooling_factor}" \
                 f"_l{latent_dim}" \
                 f"_h{hidden_dim}" \
                 f"_d{dicretize}" \
                 f"_s{scale_factor}" \
                 f"_m{model_type}" \
                 f"_shift{shift}" \
                 f"_data{data_set_type}"

    return model_name
