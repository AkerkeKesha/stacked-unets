import time
from collections import defaultdict
import numpy as np
import pickle
import matplotlib.pyplot as plt
import config
from train import train
from predict import predict
from dataloader import get_loader, split_etci_data, split_sn6_data
from utils import visualize_prediction


def load_data(dataset, max_data_points=None):
    if dataset == "etci":
        original_df, train_df, val_df, test_df = split_etci_data(max_data_points=max_data_points)
        train_loader, val_loader, test_loader = get_loader("etci", train_df, val_df, test_df)
    else:
        original_df, train_df, val_df, test_df = split_sn6_data(max_data_points=max_data_points)
        train_loader, val_loader, test_loader = get_loader("sn6", train_df, val_df, test_df)
    return original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader


def plot_metric_with_error(metric_name, metrics, level):
    level_key = f"level{level}"
    runs_data = [metrics[metric_name][run][level_key] for run in metrics[metric_name]]
    mean_values = np.mean(runs_data, axis=0)
    std_values = np.std(runs_data, axis=0)
    mean_values = mean_values.flatten()
    std_values = std_values.flatten()

    plt.errorbar(range(len(mean_values)), mean_values, yerr=std_values, capsize=5, marker='o')
    plt.title(f'Average {metric_name} for level{level} (± std)')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.show()


def plot_level_metrics(metrics, metric_name, n_levels):
    runs = list(metrics[metric_name].keys())
    levels = range(n_levels)
    mean_values = []
    std_values = []

    for level in range(n_levels):
        level_key = f"level{level}"
        all_runs_values = [metrics[metric_name][run][level_key] for run in runs]
        mean_value = np.mean(all_runs_values)
        std_value = np.std(all_runs_values)

        mean_values.append(mean_value)
        std_values.append(std_value)

    plt.errorbar(range(n_levels), mean_values, yerr=std_values, capsize=5, marker='o')
    plt.xlabel('Level')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Level')
    plt.xticks(levels)
    plt.savefig(f'{metric_name.lower()}_{config.dataset}.png', bbox_inches='tight')
    plt.show()


def visualize_examples(df, n_samples=5, n_levels=1):
    df = df.reset_index(drop=True)
    random_indices = df.sample(n=n_samples).index.tolist()
    visualize_prediction(image_indices=random_indices, df=df, n_levels=n_levels, dataset=config.dataset)


def start_stacked_unet(n_levels, max_data_points, run_key, metrics):
    original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader \
        = load_data(config.dataset, max_data_points=max_data_points)

    for level in range(n_levels):
        print(f"Level: [{level + 1} / {n_levels}]")
        level_key = f"level{level}"
        start = time.time()
        train_losses, val_losses, train_iou, val_iou, train_df, val_df \
            = train(train_loader, val_loader, train_df, val_df, level=level, run_key=run_key)
        timing = time.time() - start
        print(f"Takes {timing} seconds to train in level{level + 1}")
        final_predictions, test_df, mean_iou, avg_entropy = predict(test_loader, test_df, level=level, run_key=run_key)
        metrics_matching = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'test_iou': mean_iou,
            'train_iou': train_iou,
            'val_iou': val_iou,
            'timing': timing,
            'entropy': avg_entropy,
        }
        update_metrics(metrics, config.metrics, run_key, level_key, metrics_matching)

    np.save(f'{config.output_dir}/test_df_{run_key}.npy', test_df.to_dict(), allow_pickle=True)
    with open(f'{config.output_dir}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)


def return_default_dict():
    return defaultdict(list)


def initialize_metrics(metric_names):
    return defaultdict(return_default_dict)


def update_metrics(metrics, metric_names, run_key, level_key, computed_metrics):
    for metric_name in metric_names:
        if run_key not in metrics[metric_name]:
            metrics[metric_name][run_key] = defaultdict(list)
        if metric_name in computed_metrics:
            metrics[metric_name][run_key][level_key].append(computed_metrics[metric_name])


def calculate_stat(metric_dict, metric_name, level_key):
    if metric_name in metric_dict:
        values = [metric_dict[metric_name][run_key][level_key] for run_key in metric_dict[metric_name]]
        mean_value = np.mean(values)
        std_value = np.std(values)
        return mean_value, std_value
    return None, None


def run_experiments(runs=3, n_levels=1, max_data_points=None):
    metrics = initialize_metrics(config.metrics)
    for run in range(runs):
        print(f"Run: [{run + 1} / {runs}]")
        run_key = f"run{run}"
        start_stacked_unet(n_levels, max_data_points, run_key, metrics)

    for level in range(n_levels):
        print(f"Level: {level + 1}")
        level_key = f'level{level}'
        for metric_name in config.metrics:
            mean_val, std_val = calculate_stat(metrics, metric_name, level_key)
            if mean_val and std_val:
                print(f"Mean {metric_name}: {mean_val:.2f} +/- {std_val:.2f}")

    for metric_name in config.metrics:
        for level in range(n_levels):
            if metric_name in ['train_loss', 'val_loss', 'train_iou', 'val_iou']:
                plot_metric_with_error(metric_name, metrics, level)

    for metric_name in config.metrics:
        if metric_name in ['test_iou', 'timing', 'entropy']:
                plot_level_metrics(metrics, metric_name, n_levels)

    # visualize_examples(test_df, n_samples=5, n_levels=n_levels)























