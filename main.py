"""Neural Network Layer Order
"""
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import os

import torch
import torch.nn as nn
import numpy as np

from itertools import permutations

from src.data import get_dataloader
from src.config import load_config
from src.train import train
from src.utils import remove_layer_config
from src.visualize import visualize_results
from src.models import DenseLayerOrder, ConvLayerOrder
from src.parser import parse_tensorboard_event_files


def run_experiment_layer_order(network, layers, config):

    layer_configs = list(map(list, permutations(layers)))

    # Remove redundant layer configurations
    # (Dropout -> ReLU) = (ReLU -> Dropout)
    pattern = [nn.Dropout, nn.ReLU]
    layer_configs = remove_layer_config(layer_configs=layer_configs, pattern=pattern)

    print(f"Layer combinations: {len(layer_configs)}")
    for i, layer_config in enumerate(layer_configs):
        print(f"{i+1:02d}: {list(map(lambda x: x.__name__, layer_config))}")

    random_seed = config["random_seed"]

    # Run experiments
    for i in tqdm(range(config["n_runs"])):
        print(f"Repetition {i+1}")

        for j, layer_config in enumerate(layer_configs):
            print(f"Configuration {j + 1}")

            exp_name = "".join(list(map(lambda x: f"_{x.__name__}", layer_config)))
            config["name"] = exp_name

            # Seed random number generator
            torch.manual_seed(random_seed + i)
            np.random.seed(random_seed + i)

            # Get the data
            dataloader = get_dataloader(config=config)

            # Get the model
            model = network(layer_config=layer_config, config=config)
            model.to(config["device"])

            train(model=model, dataloader=dataloader, config=config)

            print("Finished training.\n")

    print("Experiment finished.")


def get_kwargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", choices=("train", "visualize"))
    parser.add_argument("--dataset", default="cifar10", choices=("cifar10", "fmnist", "mnist"))
    parser.add_argument("--network", default="cnn", choices=("cnn", "mlp"))
    parser.add_argument("--n_runs", default=10, type=int)
    parser.add_argument("--random_seed", default=69420, type=int)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--runs_dir", default="runs")
    kwargs = parser.parse_args()
    return kwargs


def main():

    kwargs = get_kwargs()

    # Load config
    config = load_config(file_path="config.yml")

    # Update config
    config["dataset"] = kwargs.dataset
    config["network"] = kwargs.network
    config["n_runs"] = kwargs.n_runs
    config["random_seed"] = kwargs.random_seed
    config["outputs_dir"] = os.path.join(kwargs.outputs_dir, kwargs.dataset, kwargs.network)
    config["results_dir"] = os.path.join(kwargs.results_dir, kwargs.dataset, kwargs.network)
    config["data_dir"] = os.path.join(kwargs.data_dir)
    config["runs_dir"] = os.path.join(kwargs.runs_dir, kwargs.dataset, kwargs.network)
    config["data_dir"] = kwargs.data_dir
    print(json.dumps(config, indent=4))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    print(f"Using {device}")

    if kwargs.mode == "train":

        # Create folder structure
        Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["runs_dir"]).mkdir(parents=True, exist_ok=True)

        if kwargs.network == "cnn":

            layers = [
                nn.BatchNorm2d,
                nn.Conv2d,
                nn.ReLU
            ]
            run_experiment_layer_order(network=ConvLayerOrder, layers=layers, config=config)

        elif kwargs.network == "mlp":

            layers = [
                nn.BatchNorm1d,
                nn.Linear,
                nn.ReLU
            ]
            run_experiment_layer_order(network=DenseLayerOrder, layers=layers, config=config)

        else:
            raise NotImplementedError(f"Network {kwargs.network} not implemented.")

    elif kwargs.mode == "visualize":

        # Create folder structure
        Path(config["outputs_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["results_dir"]).mkdir(parents=True, exist_ok=True)

        # Path(os.path.join(config["runs_dir"], config["dataset"])).mkdir(parents=True, exist_ok=True)

        runs_dir = os.path.join(kwargs.runs_dir, kwargs.dataset, kwargs.network)
        outputs_dir = os.path.join(kwargs.outputs_dir, kwargs.dataset, kwargs.network)
        results_dir = os.path.join(kwargs.results_dir, kwargs.dataset, kwargs.network)

        parse_tensorboard_event_files(runs_dir=runs_dir, outputs_dir=outputs_dir)
        visualize_results(outputs_dir=outputs_dir, results_dir=results_dir)

    else:
        raise NotImplementedError(f"Mode '{kwargs.mode}' not recognized.")


if __name__ == "__main__":
    main()
