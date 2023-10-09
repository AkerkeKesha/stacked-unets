import time
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
    runs_data = [metrics[metric_name][run][level] for run in metrics[metric_name]]
    mean_values = np.mean(runs_data, axis=0)
    std_values = np.std(runs_data, axis=0)
    plt.errorbar(range(len(mean_values)), mean_values, yerr=std_values, capsize=5, marker='o')
    plt.title(f'Average {metric_name} for level {level} (± std)')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.show()


def plot_level_metrics(metrics, metric_name):
    runs = list(metrics[metric_name].keys())
    levels = list(metrics[metric_name][runs[0]].keys())

    mean_values = []
    std_values = []

    for level in levels:
        all_runs_values = [metrics[metric_name][run][level] for run in runs]
        mean_value = np.mean(all_runs_values)
        std_value = np.std(all_runs_values)

        mean_values.append(mean_value)
        std_values.append(std_value)

    plt.errorbar(range(len(levels)), mean_values, yerr=std_values, capsize=5, marker='o')
    plt.xlabel('Level')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Level')
    plt.xticks(range(len(levels)), labels=levels)
    plt.show()


def visualize_examples(df, n_samples=5, n_levels=1):
    df = df.reset_index(drop=True)
    random_indices = df.sample(n=n_samples).index.tolist()
    visualize_prediction(image_indices=random_indices, df=df, n_levels=n_levels, dataset=config.dataset)


def start_stacked_unet(n_levels, max_data_points, run_key, metrics):
    for metric_name in metrics.keys():
        if run_key not in metrics[metric_name]:
            metrics[metric_name][run_key] = {}

    original_df, train_df, val_df, test_df, train_loader, val_loader, test_loader \
        = load_data(config.dataset, max_data_points=max_data_points)

    for level in range(n_levels):
        print(f"Level: [{level + 1} / {n_levels}]")
        level_key = f"level_{level}"

        start = time.time()
        train_losses, val_losses, train_iou, val_iou, train_df, val_df \
            = train(train_loader, val_loader, train_df, val_df, level=level)
        timing = time.time() - start
        print(f"Takes {timing} seconds to train in level{level + 1}")
        final_predictions, test_df, mean_iou, avg_entropy = predict(test_loader, test_df, level=level)

        for metric_name, metric_value in [('train_loss', train_losses),
                                          ('val_loss', val_losses),
                                          ('test_iou', mean_iou),
                                          ('train_iou', train_iou ),
                                          ('val_iou', val_iou),
                                          ('timing', timing),
                                          ('entropy', avg_entropy),
                                          ]:
            if level_key not in metrics[metric_name][run_key]:
                metrics[metric_name][run_key][level_key] = []
            metrics[metric_name][run_key][level_key].append(metric_value)

    np.save(f'{config.output_dir}/test_df_{run_key}.npy', test_df.to_dict(), allow_pickle=True)
    with open(f'{config.output_dir}/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    for level_key in metrics['train_loss'][run_key].keys():
        for metric_name in ['train_loss', 'val_loss', 'train_iou', 'val_iou']:
            plot_metric_with_error(metric_name, metrics, level_key)

    for metric_name in ['test_iou', 'timing', 'entropy']:
        plot_level_metrics(metrics, metric_name)

    # visualize_examples(test_df, n_samples=5, n_levels=n_levels)


def run_experiments(runs=3, n_levels=1, max_data_points=None):
    metrics = {
        'train_loss': {},
        'val_loss': {},
        'train_iou': {},
        'val_iou': {},
        'test_iou': {},
        'timing': {},
        'entropy': {},
    }
    for run in range(runs):
        print(f"Run: [{run + 1} / {runs}]")
        run_key = f"run{run}"
        start_stacked_unet(n_levels, max_data_points, run_key, metrics)

    with open(f'{config.output_dir}/metrics.pkl', 'rb') as f:
        metrics = pickle.load(f)

    for level in range(n_levels):
        level_key = f'level{level}'

        mean_iou = np.mean([metrics['test_iou'][run_key][level_key] for run_key in metrics['test_iou']])
        std_iou = np.std([metrics['test_iou'][run_key][level_key] for run_key in metrics['test_iou']])

        mean_train_loss = np.mean([metrics['train_loss'][run_key][level_key] for run_key in metrics['train_loss']])
        std_train_loss = np.std([metrics['train_loss'][run_key][level_key] for run_key in metrics['train_loss']])

        mean_val_loss = np.mean([metrics['val_loss'][run_key][level_key] for run_key in metrics['val_loss']])
        std_val_loss = np.std([metrics['val_loss'][run_key][level_key] for run_key in metrics['val_loss']])

        mean_train_iou = np.mean([metrics['train_iou'][run_key][level_key] for run_key in metrics['train_iou']])
        std_train_iou = np.std([metrics['train_iou'][run_key][level_key] for run_key in metrics['train_iou']])

        mean_val_iou = np.mean([metrics['val_iou'][run_key][level_key] for run_key in metrics['val_iou']])
        std_val_iou = np.std([metrics['val_iou'][run_key][level_key] for run_key in metrics['val_iou']])

        print(f"Level: {level + 1}")
        print(f"Mean IoU: {mean_iou:.2f} +/- {std_iou:.2f}")
        print(f"Mean Train Loss: {mean_train_loss:.2f} +/- {std_train_loss:.2f}")
        print(f"Mean Validation Loss: {mean_val_loss:.2f}  +/- {std_val_loss:.2f}")
        print(f"Mean Train IoU: {mean_train_iou:.2f} +/- {std_train_iou:.2f}")
        print(f"Mean Validation IoU: {mean_val_iou:.2f}  +/- {std_val_iou:.2f}")























