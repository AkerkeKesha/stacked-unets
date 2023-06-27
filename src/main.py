import time
import os
import numpy as np
import pandas as pd
from glob import glob
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


def save_metrics(train_iou, train_losses, val_iou, val_losses):
    metrics = ["train_losses", "val_losses", "train_iou", "val_iou"]
    new_values = [train_losses, val_losses, train_iou, val_iou]
    for metric, new_value in zip(metrics, new_values):
        metric_file = f'{config.output_dir}/{metric}_{config.dataset}.npy'
        if os.path.exists(metric_file):
            old_values = np.load(metric_file)
            combined_values = np.append(old_values, new_value)
            np.save(metric_file, combined_values)
        else:
            np.save(metric_file, new_value)
    print(f"Done saving evaluation metrics/losses on train/val")


def plot_metrics_per_level(metric_names, metric_labels, plot_filename, num_epochs, n_levels=1):
    assert len(metric_names) == len(metric_labels), "Mismatch in number of metrics and labels."
    epochs = range(1, num_epochs + 1)

    for level in range(n_levels):
        plt.figure(figsize=(10, 6))
        for metric_name, metric_label in zip(metric_names, metric_labels):
            metric_values = np.load(f'{config.output_dir}/{metric_name}_{config.dataset}.npy')
            level_metric_values = metric_values[level * num_epochs : (level + 1) * num_epochs]
            plt.plot(epochs, level_metric_values, label=f"{metric_label}")

        plt.xlabel("Epoch")
        plt.title(f"Metrics for level {level}")
        plt.legend()

        plt.savefig(f'{config.output_dir}/{plot_filename}_level{level}_{config.dataset}.png', bbox_inches='tight')
        plt.show()


def visualize_examples(df):
    for image in config.SAMPLE_IMAGES:
        visualize_prediction(image, df)


def start_basic_unet(n_levels=1, max_data_points=None):
    original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader \
        = load_data(config.dataset, max_data_points=max_data_points)
    test_mean_iou_levels = []
    timing_levels = []
    for level in range(n_levels):
        print(f"Level: [{level + 1} / {n_levels}]")
        start = time.time()
        train_losses, val_losses, train_iou, val_iou, train_df, val_df \
            = train(config.num_epochs, train_loader, val_loader, train_df, val_df, level=level)
        timing_levels.append(time.time() - start)
        save_metrics(train_iou, train_losses, val_iou, val_losses)
        final_predictions, test_df, mean_iou = predict(test_loader, test_df, level=level)
        test_mean_iou_levels.append(mean_iou)
        np.save(f'{config.output_dir}/predictions_{config.dataset}.npy',
                final_predictions,
                fix_imports=True,
                allow_pickle=False)
        updated_df = pd.concat([train_df, val_df, test_df])
        visualize_examples(updated_df)
        print(f"Finished visualizing some predictions for level {level}.")
    np.save(f'{config.output_dir}/mean_iou_levels_{config.dataset}.npy', np.array(test_mean_iou_levels))
    np.save(f'{config.output_dir}/timings_levels_{config.dataset}.npy', np.array(timing_levels))
    print(f"All levels finished")


def show_results(n_levels=1):
    for level in range(n_levels):
        plot_metrics_per_level(['train_losses', 'val_losses'],
                               ['Training Loss', 'Validation Loss'],
                               'loss_plot', config.num_epochs, level)
        plot_metrics_per_level(['train_iou', 'val_iou'],
                               ['Training Mean IoU', 'Validation Mean IoU'],
                               'iou_plot', config.num_epochs, level)












