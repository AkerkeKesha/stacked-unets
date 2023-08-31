import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.utils import grayscale_to_rgb


def compute_iou(mask_pred, mask_true):
    """
    Compute Intersection over Union (IoU) between true and predicted segmentation masks
    :param mask_pred: Predicted mask
    :param mask_true: Ground truth mask
    :return: IoU
    """
    intersection = np.logical_and(mask_true, mask_pred)
    union = np.logical_or(mask_true, mask_pred)
    iou_score = np.sum(intersection) / np.sum(union)
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


def visualize_best_worst_predictions(df, best_idx, worst_idx):
    """
    Visualize best and worst predictions
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
            semantic_map = cv2.imread(row['semantic_map_prev_level'], 0) / 255.0

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
            plt.imshow(semantic_map, cmap='jet')
            plt.title('Semantic Map')

            plt.suptitle(f"{title} - Sample {i + 1}")
            plt.show()






