import os
from collections import defaultdict
import cv2
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import config


def collect_metrics_from_files(metrics_base_names, source_dir, levels, dataset="etci"):
    """
    Collect metrics saved in npy files into a dictionary.

    @param metrics_base_names: List of base names of metrics like ["train_losses", "val_losses", "train_iou", "val_iou"]
    @param source_dir: path to the directory where npy files are located
    @param levels: List of levels like [0, 1, 2, 3]
    @param dataset: Name of the dataset (default "etci")
    @return: Dictionary containing the metrics
    """
    metrics_dict = {}

    for metric_base_name in metrics_base_names:
        metrics_dict[metric_base_name] = {}

        for level in levels:
            level_key = f"level{level}"
            file_name = f"{metric_base_name}_level{level}_{dataset}.npy"
            file_path = os.path.join(source_dir, file_name)
            another_file_name = f"{metric_base_name}_levels_{dataset}.npy"
            another_file_path = os.path.join(source_dir, another_file_name)
            if os.path.exists(file_path):
                metrics_array = np.load(file_path)
                metrics_dict[metric_base_name][level_key] = metrics_array.tolist()
            elif os.path.exists(another_file_path):
                metrics_array = np.load(another_file_path)
                metrics_dict[metric_base_name][level_key] = metrics_array.tolist()
                break
            else:
                print(f"File {file_name} does not exist.")

    return metrics_dict


def update_metrics_dict(metrics_dict, new_metric):
    for key, value in new_metric.items():
        if key not in metrics_dict:
            metrics_dict[key] = value
        else:
            metrics_dict[key].update(value)


def collect_metrics_from_single_file(metric_base_name, levels, epochs_per_level, output_dir, dataset="etci"):
    assert len(levels) == len(epochs_per_level), "Mismatch in number of levels and epochs_per_level"

    metrics_dict = {}
    file_name = f"{metric_base_name}_{dataset}.npy"
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        metrics_array = np.load(file_path)
        last_elements = metrics_array[-sum(epochs_per_level):]

        start_idx = 0
        for level, num_epochs in zip(levels, epochs_per_level):
            level_key = f"level{level}"
            end_idx = start_idx + num_epochs
            metrics_dict[level_key] = last_elements[start_idx:end_idx].tolist()
            start_idx = end_idx
    else:
        print(f"File {file_name} does not exist.")

    return {metric_base_name: metrics_dict}


def plot_metrics(metrics_dict, metric_name='Loss'):
    """
    Plot metrics like loss or IoU for each level.

    @param metrics_dict: Dictionary where keys are level identifiers and values are lists of metric values over epochs.
    @param metric_name: Name of the metric ('Loss', 'IoU', etc.)
    """
    for level, level_metrics in metrics_dict.items():
        plt.plot(level_metrics, label=f"Level {level}")

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Time')
    plt.legend()
    plt.savefig(f'{metric_name.lower()}_{config.dataset}.png', bbox_inches='tight')
    plt.show()


def read_metrics(file_path):
    """
    Read pickled metrics dict, which is a nested dict of runs and/or levels
    @param file_path: the path to metrics file
    @return: pickled metrics as dictionary
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def merge_dicts(main_dict, new_dict, run_key):
    for metric, values in new_dict.items():
        if metric not in main_dict:
            main_dict[metric] = {}

        if run_key not in main_dict[metric]:
            main_dict[metric][run_key] = defaultdict(list)

        for level, level_values in values.items():
            if isinstance(level_values, list):
                main_dict[metric][run_key][level].extend(level_values)
            elif isinstance(level_values, defaultdict):
                for sublevel, sublevel_values in level_values.items():
                    main_dict[metric][run_key][sublevel].extend(sublevel_values)


def filter_positive_flood_masks(df, flood_label_column='flood_label_path'):
    valid_rows = []
    for idx, row in df.iterrows():
        flood_label_path = row[flood_label_column]
        flood_mask = cv2.imread(flood_label_path, cv2.IMREAD_GRAYSCALE)

        if np.any(flood_mask == 255):
            valid_rows.append(idx)

    return df.loc[valid_rows]


def find_prediction_image(searched_value, df, output_type=config.output_type):
    if output_type == "semantic_map":
        mask = df["semantic_map_prev_level"].str.endswith(searched_value)
    elif output_type == "softmax_prob":
        mask = df["softmax_prob_prev_level"].str.endswith(searched_value)
    indices = df.loc[mask].index
    if len(indices) > 0:
        return indices[0]
    else:
        print(f"No match found for {searched_value} in the column")
        return None