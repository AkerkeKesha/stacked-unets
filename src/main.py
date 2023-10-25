import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import config
from src.train import train
from src.predict import predict
from src.dataloader import get_loader, split_etci_data, split_sn6_data
from src.utils import (
    visualize_prediction,
    initialize_metrics,
    update_metrics,
    calculate_stat
)


def load_data(dataset, max_data_points=None):
    if dataset == "etci":
        original_df, train_df, val_df, test_df = split_etci_data(max_data_points=max_data_points)
        train_loader, val_loader, test_loader = get_loader("etci", train_df, val_df, test_df)
    else:
        original_df, train_df, val_df, test_df = split_sn6_data(max_data_points=max_data_points)
        train_loader, val_loader, test_loader = get_loader("sn6", train_df, val_df, test_df)
    return original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader


def plot_pair_metrics_with_error(metric_names, metrics, level, labels=None, filename=None):
    level_key = f"level{level}"
    plt.figure(figsize=(10, 6))

    if not labels:
        labels = metric_names
    assert len(metric_names) == len(labels), "Mismatch in number of metrics and labels."

    for metric_name, label in zip(metric_names, labels):
        runs_data = [metrics[metric_name][run][level_key] for run in metrics[metric_name]]
        mean_values = np.mean(runs_data, axis=0)
        std_values = np.std(runs_data, axis=0)
        mean_values = mean_values.flatten()
        std_values = std_values.flatten()
        epochs = range(1, len(mean_values) + 1)

        line = plt.errorbar(epochs, mean_values, yerr=std_values, capsize=5, marker='o')
        line.set_label(f"{label}")

    plt.title(f'Train/val metrics for level{level} (± std)')
    plt.xlabel('Epoch')
    plt.legend()
    if filename:
        plt.savefig(f'{config.output_dir}/{filename}_level{level}_{config.dataset}.png', bbox_inches='tight')
    plt.show()


def plot_level_metrics(metrics, metric_name, n_levels):
    runs = list(metrics[metric_name].keys())
    mean_values = []
    std_values = []

    for level in range(n_levels):
        level_key = f"level{level}"

        all_runs_values = [v for run in runs
                           for v in metrics[metric_name][run].get(level_key, []) if v is not None]

        flat_values = [item for sublist in all_runs_values for item in
                       (sublist if isinstance(sublist, list) else [sublist])]

        if flat_values:
            mean_value = np.mean(flat_values)
            std_value = np.std(flat_values)
            mean_values.append(mean_value)
            std_values.append(std_value)
        else:
            mean_values.append(None)
            std_values.append(None)

    plt.errorbar(range(n_levels), mean_values, yerr=std_values, capsize=5, marker='o')
    plt.xlabel('Level')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Level')
    plt.xticks(range(n_levels))
    plt.savefig(f'{config.output_dir}/{metric_name.lower()}_{config.dataset}.png', bbox_inches='tight')
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

    for level in range(n_levels):
        plot_pair_metrics_with_error(['train_loss', 'val_loss'], metrics, level,
                                     labels=['train loss', 'validation loss'], filename='Loss')
        plot_pair_metrics_with_error(['train_iou', 'val_iou'], metrics, level,
                                     labels=['train IoU', 'validation IoU'], filename='IoU')
    for metric_name in config.metrics:
        if metric_name in ['test_iou', 'timing', 'entropy']:
            plot_level_metrics(metrics, metric_name, n_levels)

    loaded_dict = np.load(f'{config.output_dir}/test_df_run0.npy', allow_pickle=True).item()
    test_df = pd.DataFrame.from_dict(loaded_dict)
    test_df = test_df.reset_index(drop=True)
    visualize_examples(test_df, n_samples=5, n_levels=n_levels)























