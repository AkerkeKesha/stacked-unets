import cv2
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from src.utils import visualize_prediction


def calculate_iou_score(mask_pred, mask_true, num_classes=2, smooth=0.0001):
    """
    Compute Intersection over Union (IoU) between true and predicted segmentation masks
    @param mask_pred: Predicted mask
    @param mask_true: Ground truth mask
    @param num_classes: The number of classes for classification
    @param smooth: Small value to avoid division by zero
    @return: IoU
    """
    conf_matrix = np.zeros((num_classes, num_classes))
    mask_pred = mask_pred.ravel()
    mask_true = mask_true.ravel()
    conf_matrix += confusion_matrix(mask_true, mask_pred, labels=range(num_classes))
    iou_score = np.diag(conf_matrix) / \
                (conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - np.diag(conf_matrix) + smooth)
    return iou_score


def get_top_n_predictions(df, n, level=0):
    """
    Get the indices of top N best and worst predictions for specific level
    @param df: DataFrame that stores IoU scores in columns named like 'iou_scores_level_x'
    @param level: Level index to select IoU score column
    @param n: Number of top samples to return
    @return: Indices of top N best and worst predictions
    """
    iou_scores = df[f'iou_scores_level_{level}'].values
    best_n_idx = np.argsort(iou_scores)[-n:]
    worst_n_idx = np.argsort(iou_scores)[:n]
    return best_n_idx, worst_n_idx


def visualize_best_worst_predictions(df, best_idx, worst_idx, n_levels=1,
                                     dataset=config.dataset,
                                     output_dir=config.output_dir):
    """
    Visualize best and worst predictions depending on the given level
    @param df: dataframe that stores paths to images, ground truth masks and predicted masks
    @param best_idx: Indices of the best predictions
    @param worst_idx: Indices of the worst predictions
    @param n_levels: the number of levels to visualize
    @param output_dir: the directory where predictions are stored
    @param dataset: the dataset name, e.g. etci
    """
    for indices, title in zip([best_idx, worst_idx], ['Best Predictions', 'Worst Predictions']):
        visualize_prediction(image_indices=indices, df=df,
                             n_levels=n_levels,
                             dataset=dataset,
                             output_dir=output_dir,
                             target_filename=f'{title}_level{n_levels}.png',
                             main_title=f'{title} at Level {n_levels}')


def compare_iou_scores(df, list_of_levels):
    """
    @param df: DataFrame that contains IoU scores in columns named like 'iou_scores_level_x'
    @param list_of_levels: List of levels to compare
    @return: Dict containing improved, worsened, and constant indices for each level transition
    """
    results = {}
    for i in range(len(list_of_levels)):
        for j in range(i+1, len(list_of_levels)):
            level_a = list_of_levels[i]
            level_b = list_of_levels[j]
            delta_iou = df[f'iou_scores_level_{level_b}'] - df[f'iou_scores_level_{level_a}']

            improved_idx = np.where(delta_iou > 0)[0]
            worsened_idx = np.where(delta_iou < 0)[0]
            constant_idx = np.where(delta_iou == 0)[0]

            results[f'level_{level_a}_to_{level_b}'] = {
                'improved': improved_idx,
                'worsened': worsened_idx,
                'constant': constant_idx
            }
    return results


def plot_stacked_bar(results, list_of_levels):
    """
    @param results: Dict containing improved, worsened, and constant indices for each level transition
    @param list_of_levels: List of levels to compare
    @return: a stacked bar plot where each segment represents the difference in IoU scores
    when moving from level i to level j
    """
    plt.figure(figsize=(4, 8))
    data = pd.DataFrame(0, index=list_of_levels[:-1], columns=list_of_levels[1:], dtype=int)
    for transition, counts in results.items():
        from_level, to_level = transition.split("_to_")
        from_level = int(from_level.split("_")[-1])
        to_level = int(to_level.split("_")[-1])
        data.at[from_level, to_level] = np.round(len(counts['improved']) - len(counts['worsened']))
    bottom_pos = 0
    bottom_neg = 0
    n = len(data.columns) * len(data.index)
    colormap = plt.cm.get_cmap("tab20", n)

    for idx, row in data.iterrows():
        for jdx, value in enumerate(row):
            color = colormap(idx * len(data.columns) + jdx)
            label = f"Level {idx} to {jdx + 1}"
            if value > 0:
                plt.bar('Transitions', value, bottom=bottom_pos, color=color, label=label)
                bottom_pos += value
            elif value < 0:
                plt.bar('Transitions', value, bottom=bottom_neg, color=color, label=label)
                bottom_neg += value
    print(data)
    plt.title('IoU Score Changes Across Levels')
    plt.xlabel('Transition')
    plt.ylabel('Net Change in IoU values')
    plt.legend(title='Level Transitions', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f'{config.dataset}_stacked_bar.png', bbox_inches='tight')
    plt.show()


def plot_metrics_with_stderror(metrics, metric_name='train_loss', plot_name='Train Loss'):
    """
    Plot metrics like loss or IoU for each level over time.

    @param metrics: Nested dictionary containing the metrics data.
    @param metric_name: Name of the metric to be plotted ('train_loss', 'val_loss', etc.)
    @param plot_name: String name for plot title and label
    """
    plt.figure(figsize=(10, 6))
    n_levels = len(next(iter(metrics[metric_name].values())))

    for level in range(n_levels):
        level_key = f'level{level}'
        all_runs_level_values = [run[level_key] for run in metrics[metric_name].values()]

        all_runs_level_values = [np.array(values) for values in all_runs_level_values]

        mean_values = np.mean(all_runs_level_values, axis=0)
        std_values = np.std(all_runs_level_values, axis=0)

        mean_values = mean_values.flatten()
        std_values = std_values.flatten()

        epochs = range(len(mean_values))
        plt.errorbar(epochs, mean_values, yerr=std_values, capsize=5, marker='o', label=f'level {level}', linewidth=2,
                     markersize=6)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(plot_name.capitalize(), fontsize=12)
    plt.title(f'{plot_name.capitalize()} across epochs', fontsize=14)
    plt.legend(fontsize=10)

    plt.savefig(f'{metric_name.lower()}.png', bbox_inches='tight')
    plt.show()








