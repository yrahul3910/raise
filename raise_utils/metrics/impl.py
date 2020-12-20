from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
import math
import numpy as np


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
    _, tn, fp, fn = get_confusion_matrix(y_true, y_pred)
    return 1. * fp / (fp + tn) if fp + tn != 0 else 0


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


def get_popt20(data) -> float:
    """
    Get popt20 score.

    :param data: Pandas DataFrame with data. Must contain columns "bug", "loc", and "prediction".
    :return: popt20 score
    """
    def subtotal(x):
        xx = [0]
        for _, t in enumerate(x):
            xx += [xx[-1] + t]
        return xx[1:]

    def get_recall_(true):
        total_true = float(len([i for i in true if i == 1]))
        hit = 0.0
        recall = []
        for i, el in enumerate(true):
            if el == 1:
                hit += 1
            recall += [hit / total_true if total_true else 0.0]
        return recall

    data.sort_values(by=["bug", "loc"], ascending=[0, 1], inplace=True)
    x_sum = float(sum(data['loc']))
    x = data['loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)

    # get  AUC_optimal
    yy = get_recall_(data['bug'].values)
    xxx = [i for i in xx if i <= 0.2]
    yyy = yy[:len(xxx)]
    s_opt = round(auc(xxx, yyy), 3)

    # get AUC_worst
    xx = subtotal(x[::-1])
    yy = get_recall_(data['bug'][::-1].values)
    xxx = [i for i in xx if i <= 0.2]
    yyy = yy[:len(xxx)]
    try:
        s_wst = round(auc(xxx, yyy), 3)
    except FloatingPointError:
        s_wst = 0

    # get AUC_prediction
    data.sort_values(by=["prediction", "loc"], ascending=[0, 1], inplace=True)
    x = data['loc'].apply(lambda t: t / x_sum)
    xx = subtotal(x)
    yy = get_recall_(data['bug'].values)
    xxx = [k for k in xx if k <= 0.2]
    yyy = yy[:len(xxx)]
    try:
        s_m = round(auc(xxx, yyy), 3)
    except ValueError:
        return 0

    popt = (s_m - s_wst) / (s_opt - s_wst)
    return round(popt, 3)
