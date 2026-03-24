from model_with_spec import Model, Config
from utils.others.json_to_instance import json_to_instance
from audioencoder import Config as EncoderConfig

if __name__ == "__main__":

    save_dir = "save_dir/transfer/voxceleb1"
    config_path = f"{save_dir}/config.json"
    model_config_path = f"{save_dir}/encoder_config.json"
    config = json_to_instance(config_path, Config)
    encoder_config = json_to_instance(model_config_path, EncoderConfig)
    model = Model(encoder_config, config)

    num_params: int = 0
    for p in model.encoder.parameters():
        num_params += p.numel()

    print(num_params)