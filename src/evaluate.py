from sklearn.metrics import confusion_matrix
import numpy as np


class IntersectionOverUnion:
    def __init__(self, num_classes=2, smooth=1):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.count = 0

    def update(self, y_pred, y_true):
        y_pred = np.argmax(y_pred, axis=1).ravel()  # convert to 1D array
        y_true = y_true.ravel()  # convert to 1D array
        self.conf_matrix += confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        self.count += y_true.size

    def mean_iou(self):
        iou = np.diag(self.conf_matrix) / (self.conf_matrix.sum(axis=1) + self.conf_matrix.sum(axis=0) - np.diag(self.conf_matrix) + self.smooth)
        mean_iou = np.nanmean(iou)
        return mean_iou


