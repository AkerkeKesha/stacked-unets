import time
import math
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import config
from train import train
from predict import predict
from dataloader import get_loader, split_etci_data, split_sn6_data
from utils import plot_single_prediction, visualize_prediction, get_image_name_from_path


def load_data(dataset, n_levels=0, max_data_points=None):
    if dataset == "etci":
        original_df, train_df, val_df, test_df = split_etci_data(max_data_points=max_data_points)
        train_loader, val_loader, test_loader = get_loader("etci", train_df, val_df, test_df, n_levels=n_levels)
    else:
        original_df, train_df, val_df, test_df = split_sn6_data(max_data_points=max_data_points)
        train_loader, val_loader, test_loader = get_loader("sn6", train_df, val_df, test_df, n_levels=n_levels)
    return original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader


def save_metrics(train_iou, train_losses, val_iou, val_losses):
    np.save(f'{config.output_dir}/train_losses_{config.dataset}.npy', train_losses)
    np.save(f'{config.output_dir}/val_losses_{config.dataset}.npy', val_losses)
    np.save(f'{config.output_dir}/train_iou_{config.dataset}.npy', train_iou)
    np.save(f'{config.output_dir}/val_iou_{config.dataset}.npy', val_iou)
    print(f"Done saving evaluation metrics/losses on train/val")


def plot_metrics(metric_names, metric_labels, plot_filename, num_epochs):
    assert len(metric_names) == len(metric_labels), "Mismatch in number of metrics and labels."
    epochs = range(1, num_epochs + 1)

    for metric_name, metric_label in zip(metric_names, metric_labels):
        metric_values = np.load(f'{config.output_dir}/{metric_name}_{config.dataset}.npy')
        plt.plot(epochs, metric_values, label=metric_label)

    plt.xlabel("Epoch")
    plt.legend()

    plt.savefig(f'{config.output_dir}/{plot_filename}_{config.dataset}.png', bbox_inches='tight')
    plt.show()


def plot_predictions(test_df):
    n_batches = math.ceil(len(test_df) / config.batch_size)
    for batch in range(n_batches):
        start = batch * config.batch_size
        end = min((batch + 1) * config.batch_size, len(test_df))
        for index in range(start, end):
            df_row = test_df.iloc[index]
            vv_image_path = df_row["vv_image_path"]
            image_name = get_image_name_from_path(vv_image_path)
            semantic_map_path = df_row["semantic_map_prev_level"]
            plot_single_prediction(image_name, semantic_map_path, f"{config.output_dir}/{config.dataset}_labels")
        print(f"Finished plotting batch {batch + 1}/{n_batches}")
    print(f"All predictions finished plotting")


def visualize_results(original_df):
    labels_dir = os.path.join(config.output_dir, f'{config.dataset}_labels')
    image_ids = []
    for file_path in glob(f'{labels_dir}/*.png'):
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        # five elements to be removed from name 'semantic_map_level_0_image_'
        image_id = '_'.join(parts[5:]).split('.')[0]
        image_ids.append(image_id)
    for index in config.SAMPLE_INDICES:
        visualize_prediction(image_ids[index], original_df)


def start_basic_unet(n_levels=0, max_data_points=None):
    original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader \
        = load_data(config.dataset, n_levels=n_levels, max_data_points=max_data_points)
    start = time.time()
    train_losses, val_losses, train_iou, val_iou, train_df, val_df \
        = train(config.num_epochs, train_loader, val_loader, train_df, val_df, n_levels=n_levels)
    print(f"{time.time() - start} seconds to train")

    save_metrics(train_iou, train_losses, val_iou, val_losses)
    plot_metrics(['train_losses', 'val_losses'], ['Training Loss', 'Validation Loss'], 'loss_plot', config.num_epochs)
    plot_metrics(['train_iou', 'val_iou'], ['Training Mean IoU', 'Validation Mean IoU'], 'iou_plot', config.num_epochs)

    final_predictions, test_df = predict(test_loader, test_df, n_levels=n_levels)
    np.save(f'{config.output_dir}/predictions_{config.dataset}.npy',
            final_predictions,
            fix_imports=True,
            allow_pickle=False)
    updated_df = pd.concat([train_df, val_df, test_df])
    visualize_results(updated_df)
    print(f"Finished visualizing some predictions.")











