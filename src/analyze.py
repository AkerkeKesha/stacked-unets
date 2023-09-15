from sklearn.metrics import confusion_matrix
import numpy as np
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


def get_top_n_predictions(iou_scores, n):
    """
    Get the indices of top N best and worst predictions
    :param iou_scores: List of IoU scores for all the samples
    :param n: Number of top samples to return
    :return: Indices of top N best and worst predictions
    """
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
                             target_filename=f'{title}.png',
                             main_title=title)


def compare_iou_scores(iou_scores):
    delta_iou = iou_scores[1] - iou_scores[0]
    improved_idx = np.where(delta_iou > 0)[0]
    worsened_idx = np.where(delta_iou < 0)[0]
    constant_idx = np.where(delta_iou == 0)[0]
    return improved_idx, worsened_idx, constant_idx


def plot_losses(losses):
    """
     A faster drop in the curve for example at the 2nd level indicates quicker convergence.
    """
    plt.plot(losses[0], label="Level 1")
    plt.plot(losses[1], label="Level 2")
    plt.legend()
    plt.show()


def plot_iou(ious):
    """

    """
    plt.plot(ious[0], label="Level 1")
    plt.plot(ious[1], label="Level 2")
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








