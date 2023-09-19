import numpy as np
from src.evaluate import IntersectionOverUnion
from src.analyze import calculate_iou_score


def test_intersection_over_union():
    y_pred = np.array([[1, 0], [0, 1]]).flatten()
    y_true = np.array([[1, 0], [0, 1]]).flatten()
    metric = IntersectionOverUnion(num_classes=2)
    metric.update(y_pred, y_true)
    assert metric.mean_iou() == 1.0  # Both matrices are identical, IoU should be 1


def test_calculate_iou_score():
    mask_pred = np.array([[1, 0], [0, 1]])
    mask_true = np.array([[1, 0], [0, 1]])
    iou_score = calculate_iou_score(mask_pred, mask_true)
    assert abs(iou_score[0] - 1.0) < 1e-3
    assert abs(iou_score[1] - 1.0) < 1e-3
