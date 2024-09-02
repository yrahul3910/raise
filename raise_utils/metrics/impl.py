import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_confusion_matrix(y_true, y_pred) -> tuple:
    """
    Returns tp, tn, fp, fn

    :param y_true: True labels
    :param y_pred: Predictions
    :return: (tp, tn, fp, fn)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp, tn, fp, fn


def get_accuracy(y_true, y_pred) -> float:
    """
    Returns the accuracy score

    :param y_true: True labels
    :param y_pred: Predictions
    :return: Accuracy score
    """
    # We need to cast to np.array so that .shape exists
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = y_true.argmax(axis=1)
    if len(y_pred.shape) > 1:
        y_pred = y_pred.argmax(axis=1)

    return accuracy_score(y_true, y_pred)


def get_f1_score(y_true, y_pred) -> float:
    """
    Returns the F-1 score

    :param y_true: True labels
    :param y_pred: Predictions
    :return: F-1 score
    """
    if len(np.unique(y_true)) > 2:
        average = None
    else:
        average = 'binary'
    return f1_score(y_true, y_pred, average=average)


def get_recall(y_true, y_pred) -> float:
    """
    Returns the recall score

    :param y_true: True labels
    :param y_pred: Predictions
    :return: Recall score
    """
    if len(np.unique(y_true)) > 2:
        average = None
    else:
        average = 'binary'
    return recall_score(y_true, y_pred, average=average)


def get_precision(y_true, y_pred) -> float:
    """
    Returns the precision.

    :param y_true: True labels
    :param y_pred: Predictions
    :return: Precision
    """
    if len(np.unique(y_true)) > 2:
        average = None
    else:
        average = 'binary'
    return precision_score(y_true, y_pred, average=average)


def get_pf(y_true, y_pred) -> float:
    """
    Returns the false alarm rate

    :param y_true: True labels
    :param y_pred: Predictions
    :return: False alarm rate
    """
    _, tn, fp, _ = get_confusion_matrix(y_true, y_pred)
    return 1. * fp / (fp + tn) if fp + tn != 0 else 0


def get_pd_pf(y_true, y_pred) -> float:
    """
    Returns the value of recall - false alarm rate.

    :param y_true: True labels
    :param y_pred: Predictions
    :return: Recall - false alarm rate
    """
    return get_recall(y_true, y_pred) - get_pf(y_true, y_pred)


def get_roc_auc(y_true, y_pred) -> float:
    """
    Returns the area under the pd/pf curve

    :param y_true: True labels
    :param y_pred: Predictions
    :return: AUC score
    """
    return roc_auc_score(y_true, y_pred)


def get_d2h(y_true, y_pred) -> float:
    """
    Returns the distance to heaven metric

    :param y_true: True labels
    :param y_pred: Predictions
    :return: d2h score
    """
    return 1. / math.sqrt(2) - math.sqrt(get_pf(y_true, y_pred) ** 2 + (1. - get_recall(y_true, y_pred)) ** 2) / math.sqrt(2)


def get_d2h2(y_true, y_pred) -> float:
    """
    Returns the distance to heaven metric

    :param y_true: True labels
    :param y_pred: Predictions
    :return: d2h score
    """
    return 1. / math.sqrt(2) - math.sqrt(2.*get_pf(y_true, y_pred) ** 2 + (1. - get_recall(y_true, y_pred)) ** 2) / math.sqrt(2)


def get_ifa(y_true, y_pred) -> float:
    ifa = 0
    actual_results = np.asarray(y_true)
    predicted_results = np.asarray(y_pred)
    index = 0
    for i, j in zip(actual_results, predicted_results):
        if ((i == "yes") and (j == "yes")) or ((i == 1) and (j == 0)):
            break
        elif ((i == "no") and (j == "yes")) or ((i == 0) and (j == 1)):
            ifa += 1
        index += 1
    return ifa


def get_g1_score(y_true, y_pred) -> float:
    """
    Returns the G-1 score

    :param y_true: True labels
    :param y_pred: Predictions
    :return: G-1 score
    """
    tp, tn, fp, fn = get_confusion_matrix(y_true, y_pred)
    pf = 1. * fp / (fp + tn) if fp + tn != 0 else 0
    recall = 1. * tp / (tp+fn) if tp + fn != 0 else 0
    g_score = (2 * recall * (1 - pf)) / (recall + 1 - pf) if recall + 1 - pf != 0 else 0
    return g_score
