import numpy as np
import cv2
import matplotlib.pyplot as plt

import config
from utils import get_image_name_from_path
from src.utils import grayscale_to_rgb
from src.evaluate import IntersectionOverUnion


def compute_mean_iou(mask_pred, mask_true):
    """
    Compute Intersection over Union (IoU) between true and predicted segmentation masks
    :param mask_pred: Predicted mask
    :param mask_true: Ground truth mask
    :return: IoU
    """
    iou_metric = IntersectionOverUnion(num_classes=2)
    iou_metric.update(mask_pred, mask_true)
    iou_score = iou_metric.mean_iou()
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


def visualize_best_worst_predictions(df, best_idx, worst_idx, level=0):
    """
    Visualize best and worst predictions depending on the given level
    :param df: dataframe that stores paths to images, ground truth masks and predicted masks
    :param best_idx: Indices of best predictions
    :param worst_idx: Indices of worst predictions
    """
    for indices, title in zip([best_idx, worst_idx], ['Best Predictions', 'Worst Predictions']):
        for i, idx in enumerate(indices):
            row = df.iloc[idx]

            vv_image = cv2.imread(row['vv_image_path'], 0) / 255.0
            vh_image = cv2.imread(row['vh_image_path'], 0) / 255.0
            rgb_image = grayscale_to_rgb(vv_image, vh_image)

            flood_label = cv2.imread(row['flood_label_path'], 0) / 255.0
            water_body_label = cv2.imread(row['water_body_label_path'], 0) / 255.0
            image_path = row["vv_image_path"]
            image = get_image_name_from_path(image_path)
            semantic_map_path = f"{config.labels_dir}/semantic_map_level_{level}_image_{image}.png"
            semantic_map = cv2.imread(semantic_map_path, 0) / 255.0

            plt.figure(figsize=(20, 10))

            plt.subplot(1, 4, 1)
            plt.imshow(rgb_image)
            plt.title('Rgb Image')

            plt.subplot(1, 4, 2)
            plt.imshow(flood_label, cmap='gray')
            plt.title('Flood Label')

            plt.subplot(1, 4, 3)
            plt.imshow(water_body_label, cmap='gray')
            plt.title('Water Body Label')

            plt.subplot(1, 4, 4)
            plt.imshow(semantic_map, cmap='gray')
            plt.title(f'Predicted on level{level}')

            plt.suptitle(f"{title} - Sample {i + 1}")
            plt.show()


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








