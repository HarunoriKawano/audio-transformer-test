import os
import argparse

import torch
import pandas as pd

from model import Config, Model
from train_framework import TrainFramework, LearningSettings
from dataset import DatasetParams, get_dataset
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

    save_dir = "save_dir/without_spectrum/gsc_v1"
    config_path = f"{save_dir}/config.json"
    model_config_path = f"{save_dir}/encoder_config.json"
    ls_path = f"{save_dir}/learning_settings.json"
    dp_path = f"{save_dir}/dataset_params.json"

    config = json_to_instance(config_path, Config)
    encoder_config = json_to_instance(model_config_path, EncoderConfig)
    ls = json_to_instance(ls_path, LearningSettings)
    dp = json_to_instance(dp_path, DatasetParams)

    results_list = []
    random_seed_list = [42, 1234, 2026, 27182, 31415]

    seed = 42
    save_seed_dir = os.path.join(save_dir, f"seed_{seed}")
    set_random_seed(seed)
    accuracy_list = []
    model = Model(encoder_config, config)
    model.encoder.load_state_dict(torch.load("save_dir/trained_weights/encoder_weight.pth"))
    preprocessor = Preprocessor(encoder_config)

    train_dataset, eval_dataset, test_dataset = get_dataset(dp)
    trainer = TrainFramework(model, ls, train_dataset, eval_dataset)

    os.makedirs(save_seed_dir, exist_ok=True)
    trainer.train(save_seed_dir)
    trainer.model.load_state_dict(torch.load(os.path.join(save_seed_dir, "best_model.pth")))
    test_loss, acc = trainer.test(test_dataset)
    results_list.append({"seed": seed, "loss": test_loss, "accuracy": acc})

    df = pd.DataFrame(results_list)
    df.to_csv(f"{save_dir}/results.csv", index=False)


