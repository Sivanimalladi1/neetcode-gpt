import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        ep = 1e-7 
        p = np.clip(y_pred, ep, 1-ep)
        ans = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1-p))
        return round(ans, 4)

        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        ep = 1e-7
        p = np.clip(y_pred, ep, 1-ep)
        p = y_true * np.log(p)
        ans = -np.mean(np.sum(p, axis=1))
        return round(ans, 4)
