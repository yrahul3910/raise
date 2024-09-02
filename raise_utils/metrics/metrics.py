import numpy as np

from raise_utils.metrics.impl import (
    get_accuracy,
    get_confusion_matrix,
    get_d2h,
    get_d2h2,
    get_f1_score,
    get_g1_score,
    get_ifa,
    get_pd_pf,
    get_pf,
    get_precision,
    get_recall,
    get_roc_auc,
)

name_map = {
    "accuracy": get_accuracy,
    "pf": get_pf,
    "d2h2": get_d2h2,
    "pd": get_recall,
    "recall": get_recall,
    "auc": get_roc_auc,
    "d2h": get_d2h,
    "ifa": get_ifa,
    "f1": get_f1_score,
    "prec": get_precision,
    "g1": get_g1_score,
    "pd-pf": get_pd_pf,
    "conf": get_confusion_matrix
}


class Metric:
    """Base class for all metrics"""

    def __init__(self, y_true, y_pred):
        """
        Initializes the Metric object

        :param y_true: True targets
        :param y_pred: Predictions
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.data = None
        self.metrics = []

    def add_metric(self, metric: str) -> None:
        """
        Adds a metric to the object

        :param metric:
        :return: None
        """
        if metric not in name_map.keys():
            raise ValueError("Invalid metric name.")

        self.metrics.append(metric)

    def add_metrics(self, metrics: list) -> None:
        """
        Adds a list of metrics.

        :param metrics: List
        :return: None
        """
        for metric in metrics:
            try:
                self.add_metric(metric)
            except ValueError | AssertionError:
                continue


class ClassificationMetrics(Metric):
    """Handles classification metrics"""

    def get_metrics(self) -> list:
        """
        Returns all metric values

        :return: List of all added metrics
        """
        self.y_pred = np.array(self.y_pred).squeeze()

        metrics = [name_map[metric](self.y_true, self.y_pred) for metric in self.metrics]

        return [metric.tolist() if isinstance(metric, np.ndarray) else metric for metric in metrics]
