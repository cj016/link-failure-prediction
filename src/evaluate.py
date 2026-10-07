# evaluate.py
# test accuracy (Eq.6) and some extra metrics from the syllabus

import numpy as np


def test_accuracy(y_true, y_pred):
    # (1/n) * sum( 1{y_pred_i = y_true_i} )
    return np.mean(y_pred == y_true)


def confusion(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def precision_recall(y_true, y_pred):
    tp, tn, fp, fn = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall
