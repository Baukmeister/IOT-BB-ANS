from util.experiment_params import Params


def _vae_model_name(
        model_folder,
        dicretize,
        hidden_dim,
        latent_dim,
        pooling_factor,
        scale_factor,
        model_type,
        shift,
        data_set_type=""
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


def vae_model_name(params: Params):
    return _vae_model_name(
        model_folder=f"../models/trained_models/{params.data_set_name}",
        dicretize=params.discretize,
        hidden_dim=params.hidden_dim,
        latent_dim=params.latent_dim,
        pooling_factor=params.pooling_factor,
        scale_factor=params.scale_factor,
        model_type=params.model_type,
        shift=params.shift,
        data_set_type=params.metric
    )


def input_dim(params: Params) -> int:
    if params.data_set_name == "simple":
        n_features = int(1 * params.pooling_factor)
    elif params.data_set_name == "household":
        if params.metric == "all":
            n_features = 7 * int(params.pooling_factor)
        else:
            n_features = 1 * int(params.pooling_factor)
    elif params.data_set_name == "wisdm":
        n_features = 3 * int(params.pooling_factor)
    elif params.data_set_name == "intel":
        if params.metric == "all":
            n_features = 4 * int(params.pooling_factor)
        else:
            n_features = 1 * int(params.pooling_factor)
    else:
        raise AttributeError(f"Data set type {params.data_set_name} is not defined!")

    return n_features
