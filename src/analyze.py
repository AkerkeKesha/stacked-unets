from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import visualize_prediction


def calculate_iou_score(mask_pred, mask_true, num_classes=2, smooth=0.0001):
    """
    Compute Intersection over Union (IoU) between true and predicted segmentation masks
    :param mask_pred: Predicted mask
    :param mask_true: Ground truth mask
    :param num_classes: The number of classes for classification
    :param smooth: Small value to avoid division by zero
    :return: IoU
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
    :param df: DataFrame that stores IoU scores in columns named like 'iou_scores_level_x'
    :param level: Level index to select IoU score column
    :param n: Number of top samples to return
    :return: Indices of top N best and worst predictions
    """
    iou_scores = df[f'iou_scores_level_{level}'].values
    best_n_idx = np.argsort(iou_scores)[-n:]
    worst_n_idx = np.argsort(iou_scores)[:n]
    return best_n_idx, worst_n_idx


def visualize_best_worst_predictions(df, best_idx, worst_idx, n_levels=1):
    """
    Visualize best and worst predictions depending on the given level
    :param df: dataframe that stores paths to images, ground truth masks and predicted masks
    :param best_idx: Indices of the best predictions
    :param worst_idx: Indices of the worst predictions
    :param n_levels: the number of levels to visualize
    """
    for indices, title in zip([best_idx, worst_idx], ['Best Predictions', 'Worst Predictions']):
        visualize_prediction(image_indices=indices, df=df,
                             n_levels=n_levels,
                             target_filename=f'{title}_level{n_levels}.png',
                             main_title=f'{title} at Level {n_levels}')


def compare_iou_scores(df, list_of_levels):
    """
    :param df: DataFrame that contains IoU scores in columns named like 'iou_scores_level_x'
    :param list_of_levels: List of levels to compare
    :return: Dict containing improved, worsened, and constant indices for each level transition
    """
    results = {}
    for i in range(len(list_of_levels) - 1):
        level_a = list_of_levels[i]
        level_b = list_of_levels[i + 1]
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


def plot_heatmap(results, list_of_levels):
    """
    Plots the heatmap where each cell (i, j) represents the difference in IoU scores when moving from level i to level j
    @param results: Dict containing improved, worsened, and constant indices for each level transition
    @param list_of_levels: List of levels to compare
    @return: heatmap representing the change in IoU scores between different levels.
    """
    data = pd.DataFrame(0, index=list_of_levels[:-1], columns=list_of_levels[1:], dtype=int)
    for transition, counts in results.items():
        from_level, to_level = transition.split("_to_")
        from_level = int(from_level.split("_")[-1])
        to_level = int(to_level.split("_")[-1])
        data.at[from_level, to_level] = np.round(len(counts['improved']) - len(counts['worsened']))

    sns.heatmap(data, annot=True, fmt="d", cmap="coolwarm")

    plt.title('IoU Score Changes Across Levels')
    plt.xlabel('To Level')
    plt.ylabel('From Level')
    plt.show()


def plot_metrics(metrics, metric_name='Loss', levels=None):
    """
    Plot metrics like loss or IoU for each level.
    :param metrics: List of metric values for each level. Each element should be list of metric values over time.
    :param metric_name: Name of the metric ('Loss', 'IoU', etc.)
    :param levels: Names or identifiers for each level.
    """
    if levels is None:
        levels = [str(i + 1) for i in range(len(metrics))]

    for i, level_metrics in enumerate(metrics):
        plt.plot(level_metrics, label=f"Level {levels[i]}")

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Over Time')
    plt.legend()
    plt.show()


def distort_semantic_maps(semantic_maps):
    """
    might have to run a new prediction after distorting the semantic input.
    If you have stored the semantic maps, you can load them, distort them, and run just the prediction again.
    iou_drop = df_test['iou_scores'] - df_test_distorted['iou_scores']
    """
    # Distortion logic: put random values to semantic map
    distorted_semantic_maps = []
    return distorted_semantic_maps








