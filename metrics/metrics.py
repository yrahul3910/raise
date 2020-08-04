from metrics.impl import *


name_map = {
    "accuracy": get_accuracy,
    "popt20": get_popt20,
    "pf": get_pf,
    "pd": get_recall,
    "recall": get_recall,
    "auc": get_roc_auc,
    "d2h": get_d2h,
    "f1": get_f1_score,
    "prec": get_precision
}


class Metric:
    """
    Base class for all metrics
    """
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


class ClassificationMetrics(Metric):
    """
    Handles classification metrics
    """
    def add_data(self, data) -> None:
        """
        Adds data for the popt20 metric
        :param data: Pandas DataFrame. Must include the columns "bug", "loc", and "prediction"
        :return: None
        """
        self.data = data

    def add_metric(self, metric: str) -> None:
        """
        Adds a metric to the object
        :param metric:
        :return: None
        """
        if metric not in name_map.keys():
            raise ValueError("Invalid metric name.")

        if metric == "popt20" and self.data is None:
            raise AssertionError("Please call add_data if including popt20 as a metric.")

        self.metrics.append(metric)

    def add_metrics(self, metrics: list) -> None:
        """
        Adds a list of metrics.
        :param metrics: List
        :return: None
        """
        for metric in metrics:
            self.add_metric(metric)

    def get_metrics(self) -> list:
        """
        Returns all metric values
        :return: List of all added metrics
        """
        metrics = [name_map[metric](self.y_true, self.y_pred) for metric in self.metrics if metric != 'popt20']
        if "popt20" in self.metrics:
            metrics.insert(self.metrics.index("popt20"), get_popt20(self.data))
        return metrics
