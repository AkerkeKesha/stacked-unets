import time
import numpy as np
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


def save_metrics(train_iou, train_losses, val_iou, val_losses, level):
    metrics = ["train_losses", "val_losses", "train_iou", "val_iou"]
    new_values = [train_losses, val_losses, train_iou, val_iou]
    for metric, new_value in zip(metrics, new_values):
        metric_file = f'{config.output_dir}/{metric}_level{level}_{config.dataset}.npy'
        np.save(metric_file, new_value)
    print(f"Done saving evaluation metrics/losses on train/val")


def plot_metrics_per_level(metric_names, metric_labels, plot_filename, level):
    assert len(metric_names) == len(metric_labels), "Mismatch in number of metrics and labels."
    plt.figure(figsize=(10, 6))
    for metric_name, metric_label in zip(metric_names, metric_labels):
        metric_file = f'{config.output_dir}/{metric_name}_level{level}_{config.dataset}.npy'
        metric_values = np.load(metric_file)
        epochs = range(1, len(metric_values) + 1)
        plt.plot(epochs, metric_values, label=f"{metric_label}")

    plt.xlabel("Epoch")
    plt.title(f"Metrics for level {level}")
    plt.legend()

    plt.savefig(f'{config.output_dir}/{plot_filename}_level{level}_{config.dataset}.png', bbox_inches='tight')
    plt.show()


def visualize_examples(df, n_samples=5, n_levels=1):
    df = df.reset_index(drop=True)
    random_indices = df.sample(n=n_samples).index.tolist()
    visualize_prediction(image_indices=random_indices, df=df, n_levels=n_levels, dataset=config.dataset)


def start_stacked_unet(n_levels, max_data_points, run,
                       all_test_mean_ious,
                       all_train_losses,
                       all_val_losses):
    original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader \
        = load_data(config.dataset, max_data_points=max_data_points)
    test_mean_ious = []
    timing_levels = []
    for level in range(n_levels):
        print(f"Level: [{level + 1} / {n_levels}]")
        start = time.time()
        train_losses, val_losses, train_iou, val_iou, train_df, val_df \
            = train(train_loader, val_loader, train_df, val_df, level=level)
        timing_levels.append(time.time() - start)
        print(f"Takes {time.time() - start} seconds to train in level{level + 1}")
        save_metrics(train_iou, train_losses, val_iou, val_losses, level=level)
        final_predictions, test_df, mean_iou = predict(test_loader, test_df, level=level)
        test_mean_ious.append(mean_iou)
        np.save(f'{config.output_dir}/predictions_{config.dataset}_run{run}_level{level}.npy',
                final_predictions,
                fix_imports=True,
                allow_pickle=False)

        if level not in all_test_mean_ious:
            all_test_mean_ious[level] = []
        if level not in all_train_losses:
            all_train_losses[level] = []
        if level not in all_val_losses:
            all_val_losses[level] = []

        all_test_mean_ious[level].append(test_mean_ious)
        all_train_losses[level].append(train_losses)
        all_val_losses[level].append(val_losses)

        np.save(f'{config.output_dir}/test_mean_iou_run{run}_level{level}_{config.dataset}.npy', mean_iou)
        np.save(f'{config.output_dir}/train_losses_run{run}_level{level}_{config.dataset}.npy', train_losses)
        np.save(f'{config.output_dir}/val_losses_run{run}_level{level}_{config.dataset}.npy', val_losses)

    np.save(f'{config.output_dir}/mean_iou_run{run}_{config.dataset}.npy', np.array(test_mean_ious))
    np.save(f'{config.output_dir}/timings_run{run}_{config.dataset}.npy', np.array(timing_levels))
    np.save(f'{config.output_dir}/test_df_run{run}.npy', test_df.to_dict(), allow_pickle=True)

    show_results(n_levels=n_levels, run=run)
    visualize_examples(test_df, n_samples=5, n_levels=n_levels)


def show_results(n_levels, run):
    for level in range(n_levels):
        plot_metrics_per_level(['train_losses', 'val_losses'],
                               ['Training Loss', 'Validation Loss'],
                               'loss_plot', level)
        plot_metrics_per_level(['train_iou', 'val_iou'],
                               ['Training Mean IoU', 'Validation Mean IoU'],
                               'iou_plot', level)

    mean_iou_levels = np.load(f'{config.output_dir}/mean_iou_run{run}_{config.dataset}.npy')
    timing_levels = np.load(f'{config.output_dir}/timings_run{run}_{config.dataset}.npy')

    # Plotting mean IoUs
    plt.figure(figsize=(10, 6))
    levels = range(n_levels)
    plt.plot(levels, mean_iou_levels, marker='o')
    plt.xlabel("Level")
    plt.ylabel("Mean IoU")
    plt.title("Mean IoU per Level")
    plt.xticks(levels)
    plt.savefig(f'{config.output_dir}/mean_iou_plot_{config.dataset}.png', bbox_inches='tight')
    plt.show()

    # Plotting timing levels
    plt.figure(figsize=(10, 6))
    plt.plot(levels, timing_levels, marker='o')
    plt.xlabel("Level")
    plt.ylabel("Time (seconds)")
    plt.title("Timing per Level")
    plt.xticks(levels)
    plt.savefig(f'{config.output_dir}/timing_plot_{config.dataset}.png', bbox_inches='tight')
    plt.show()
    print("Done plotting results")


def run_experiments(runs=3, n_levels=1, max_data_points=None):
    all_test_mean_ious = {}
    all_train_losses = {}
    all_val_losses = {}
    for run in range(runs):
        print(f"Run: [{run + 1} / {runs}]")
        start_stacked_unet(n_levels, max_data_points, run,
                           all_test_mean_ious,
                           all_train_losses,
                           all_val_losses)

    for level in range(n_levels):
        mean_iou = np.mean(all_test_mean_ious[level], axis=0)
        std_iou = np.std(all_test_mean_ious[level], axis=0)

        mean_train_loss = np.mean(all_train_losses[level], axis=0)
        std_train_loss = np.std(all_train_losses[level], axis=0)

        mean_val_loss = np.mean(all_val_losses[level], axis=0)
        std_val_loss = np.std(all_val_losses[level], axis=0)

        print(f"Level: {level + 1}")
        print(f"Mean IoU: {mean_iou:.2f}, Std Dev IoU: {std_iou:.2f}")
        print(f"Mean Train Loss: {mean_train_loss:.2f}, Std Dev Train Loss: {std_train_loss:.2f}")
        print(f"Mean Validation Loss: {mean_val_loss:.2f}, Std Dev Validation Loss: {std_val_loss:.2f}")

        np.save(f'{config.output_dir}/mean_iou_level{level}_{config.dataset}.npy', mean_iou)
        np.save(f'{config.output_dir}/std_iou_level{level}_{config.dataset}.npy', std_iou)

        np.save(f'{config.output_dir}/mean_train_loss_level{level}_{config.dataset}.npy', mean_train_loss)
        np.save(f'{config.output_dir}/std_train_loss_level{level}_{config.dataset}.npy', std_train_loss)

        np.save(f'{config.output_dir}/mean_val_loss_level{level}_{config.dataset}.npy', mean_val_loss)
        np.save(f'{config.output_dir}/std_val_loss_level{level}_{config.dataset}.npy', std_val_loss)






















