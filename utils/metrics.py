import numpy as np

def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.sum(y_true == y_pred) / len(y_true)
