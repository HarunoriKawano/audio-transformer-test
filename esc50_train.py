import os
import argparse

import torch
import pandas as pd

from model_with_spec import Config, Model
from train_framework import TrainFramework, LearningSettings
from dataset import ESC50DatasetParams, get_esc50_dataset
from utils.others.json_to_instance import json_to_instance
from utils.others.seed_setting import set_random_seed
from audioencoder import Config as EncoderConfig, Preprocessor


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--master_addr', type=str, default="localhost")
    parser.add_argument('--master_port', type=str, default="23456")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    save_dir = "save_dir/transfer/esc50"
    config_path = f"{save_dir}/config.json"
    model_config_path = f"{save_dir}/encoder_config.json"
    ls_path = f"{save_dir}/learning_settings.json"
    dp_path = f"{save_dir}/dataset_params.json"

    config = json_to_instance(config_path, Config)
    encoder_config = json_to_instance(model_config_path, EncoderConfig)
    ls = json_to_instance(ls_path, LearningSettings)
    dp = json_to_instance(dp_path, ESC50DatasetParams)

    results_list = []
    random_seed_list = [42, 1234, 2025, 27182, 31415]

    seed = 42
    save_seed_dir = os.path.join(save_dir, f"seed_{seed}")
    set_random_seed(seed)
    accuracy_list = []
    for i in range(5):
        model = Model(encoder_config, config)
        model.encoder.encoder.load_state_dict(torch.load("save_dir/trained_weights/encoder_weight.pth"))

        train_dataset, eval_dataset = get_esc50_dataset(dp, target_index=i)
        trainer = TrainFramework(model, ls, train_dataset, eval_dataset)

        os.makedirs(os.path.join(save_seed_dir, f"block{i}"), exist_ok=True)
        acc = trainer.train(os.path.join(save_seed_dir, f"block{i}"))

        accuracy_list.append(acc)
    accuracy = sum(accuracy_list) / len(accuracy_list)
    results_list.append({"seed": seed, "accuracy": accuracy})

    df = pd.DataFrame(results_list)
    df.to_csv(f"{save_dir}/results.csv", index=False)


