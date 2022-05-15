"""Script with methods to load and visualize tensorboard record files.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def path_to_key(path: str) -> str:
    key = path.split("__")[-1].split(".")[-2]
    return key


def visualize_results(outputs_dir: str, results_dir: str):
    """
    """
    # Get paths to all event files from all runs
    event_paths = glob.glob(os.path.join(outputs_dir, "*"))

    # Create folder for results
    # os.makedirs(results_dir, exist_ok=True)

    # Create dictionary for data with unique keys
    keys = set()
    for event_path in event_paths:
        key = path_to_key(path=event_path)
        keys.add(key)
    keys = list(keys)
    events = {key: list() for key in keys}
    keys.sort()

    # Link every key to unique color
    cmap = plt.get_cmap("tab20").colors
    key_to_cmap = {key: rgb for key, rgb in zip(keys, cmap)}

    # Load data and add it to dictionary
    for event_path in event_paths:
        key = path_to_key(path=event_path)
        data_dict = pd.read_pickle(event_path).to_dict()
        events[key].append(data_dict)

    # Metrics of interest
    # metrics = ["test_accuracy", "test_loss", "train_accuracy", "train_loss"]
    metrics = ["test_accuracy", "train_accuracy"]

    for metric in metrics:

        # Get length of time series
        time_series_length = list()
        for key, runs in events.items():
            for run in runs:
                time_series_length.append(len(run[metric]["step"]))
        time_series_max_length = max(time_series_length)

        # Extract time series of interest
        data = {key: list() for key in keys}
        for key, runs in events.items():
            for run in runs:
                if len(run[metric]["value"]) == time_series_max_length:
                    data[key].append(run[metric])

        # Preprocess data
        for key, value in data.items():
            x = value[0]["step"]
            y = np.stack([item["value"] for item in value])
            data[key] = dict(step=x, mean=np.mean(y, axis=0), std=np.std(y, axis=0))

        # Sort dictionary holing results by mean value of metric
        data = dict(sorted(data.items(), key=lambda item: item[-1]["mean"][-1], reverse=True))

        # Print results to console
        print_results(data=data, metric=metric)

        # Visualize data
        visualize(data=data, results_dir=results_dir, metric=metric, key_to_cmap=key_to_cmap)


def print_results(data: dict, metric: str) -> None:
    """Prints results to console.

    """
    print(f"\n| Layer configuration | {metric.replace('_', ' ').capitalize()} |")
    print(f"| --- | --- |")
    for i, (key, value) in enumerate(data.items()):
        print(f"| {key.replace('_', ' -> ')} "
              f"| {data[key]['mean'][-1]:.4f} +- {data[key]['std'][-1]:.4f} |")


def visualize(data: dict, results_dir: str, metric: str, key_to_cmap: dict) -> None:
    """Visualize data from runs.

    Args:
        data:
        results_dir:
        metric:
        key_to_cmap:

    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))

    mu_min = np.inf
    mu_max = -np.inf

    for key, value in data.items():
        x, mu, sigma = value["step"], value["mean"], value["std"]
        mu_min = min(mu_min, mu[0])
        mu_max = max(mu_max, mu[-1])
        ax.plot(x, mu, marker="o", linestyle="--", linewidth=0.5,
                label=key.replace('_', r'$\rightarrow$'), color=key_to_cmap[key])
        ax.fill_between(x, mu - sigma, mu + sigma, color=key_to_cmap[key], alpha=0.1)

    # Magnify last points
    rho = 0.2
    x0, y0 = 0.5, 0.04
    width, height = 1.0 - x0 - 0.02, 0.42
    ax2 = ax.inset_axes([x0, y0, width, height])
    for key, value in data.items():
        x, mu, sigma = value["step"], value["mean"], value["std"]
        n_points = int(rho * len(mu))
        n_points = n_points if n_points > 2 else 3
        x = x[-n_points:]
        mu = mu[-n_points:]
        sigma = sigma[-n_points:]
        ax2.plot(x, mu, marker="o", linestyle="--", linewidth=0.5, color=key_to_cmap[key])
        ax2.fill_between(x, mu - sigma, mu + sigma, color=key_to_cmap[key], alpha=0.1)
        ax2.set_facecolor((0, 0, 0, 0))
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.grid("True", linestyle="--", linewidth=0.2)
    ax.indicate_inset_zoom(ax2, linewidth=1)

    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(metric.replace("_", " ").capitalize())
    ax.set_ylim([max(0.5, 0.95*mu_min), 1.05*mu_max])
    plt.legend()
    plt.savefig(os.path.join(results_dir, f"{metric}.png"), dpi=120, transparent=True)
    plt.close(fig)
