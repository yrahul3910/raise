from raise_utils.metrics.impl import get_accuracy
from raise_utils.metrics.impl import get_popt20
from raise_utils.metrics.impl import get_pf
from raise_utils.metrics.impl import get_recall
from raise_utils.metrics.impl import get_roc_auc
from raise_utils.metrics.impl import get_d2h, get_d2h2
from raise_utils.metrics.impl import get_f1_score
from raise_utils.metrics.impl import get_precision
from raise_utils.metrics.impl import get_pd_pf
from raise_utils.metrics.impl import get_ifa
from raise_utils.metrics.impl import get_g1_score
from raise_utils.metrics.impl import get_confusion_matrix
import numpy as np


name_map = {
    "accuracy": get_accuracy,
    "popt20": get_popt20,
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

        if metric == "popt20" and self.data is None:
            raise AssertionError(
                "Please call add_data if including popt20 as a metric.")

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

    def add_data(self, data) -> None:
        """
        Adds data for the popt20 metric

        :param data: Pandas DataFrame. Must include the columns "bug", "loc", and "prediction"
        :return: None
        """
        self.data = data

    def get_metrics(self) -> list:
        """
        Returns all metric values

        :return: List of all added metrics
        """
        self.y_pred = np.array(self.y_pred).squeeze()

        metrics = [name_map[metric](self.y_true, self.y_pred) for metric in self.metrics
                   if metric != 'popt20']
        if "popt20" in self.metrics:
            metrics.insert(self.metrics.index("popt20"), get_popt20(self.data))

        return [metric.tolist() if isinstance(metric, np.ndarray) else metric for metric in metrics]
