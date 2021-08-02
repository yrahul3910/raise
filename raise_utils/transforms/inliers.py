import numpy as np


class OutlierRemoval:
    """
    Removes outliers using L1 distance [1]

    [1] Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001, January).
        On the surprising behavior of distance metrics in high dimensional
        space. In International conference on database theory (pp. 420-434).
        Springer, Berlin, Heidelberg.
    """
    def __init__(self):
        self.detector = None
        self.n_neighbors = 0
        self.vars = np.ndarray([])

    def fit_transform(self, x_train, y_train):
        """
        Initializes the object

        :param x_train: Train data
        :param y_train: Train labels
        """
        self.vars = np.array([np.amax(x_train[i]) - np.amin(x_train[i]) for i in x_train])
        idx = np.where(np.all(0.01 * self.vars < x_train) and np.all(x_train < 0.95 * self.vars))[0]
        return np.array(x_train)[idx], np.array(y_train)[idx]

    def transform(self, x_test) -> np.ndarray:
        """
        Remove outliers from data

        :param x_test: Test data
        :return: np.ndarray
        """
        x_test = np.array(x_test)
        return x_test[np.where(np.all(0.01 * self.vars < x_test) and np.all(x_test < 0.95 * self.vars))[0]]
