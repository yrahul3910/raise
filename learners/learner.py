import pandas as pd
import numpy as np


class Learner:
    """
    The base Learner class.
    """

    def __init__(self, name: str = "rf", random: bool = False, id: str = None):
        """
        Initializes a Learner object
        :param name: The name of the learner. Must be a recognized name.
        :param random: Whether to initialize the hyperparameters randomly.
        :param id: A user-identifiable id. Not necessary, just for debugging.
        """
        self.name = name
        self.random = random
        self.id = id
        self.learner = None
        self.__name__ = name
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def set_data(self, x_train, y_train, x_test, y_test) -> None:
        """
        Sets the data of the learner
        :param x_train: Training data
        :param y_train: Training labels
        :param x_test: Test data
        :param y_test: Test labels
        :return: None
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def _check_data(self) -> None:
        """
        Ensures data is set
        :return: None
        """
        assert (
            self.x_train is not None and
            self.y_train is not None and
            self.x_test is not None and
            self.y_test is not None
        ) and (
            type(self.x_train) in [np.ndarray, pd.DataFrame] and
            type(self.x_test) in [np.ndarray, pd.DataFrame] and
            type(self.y_train) in [np.ndarray, pd.DataFrame] and
            type(self.y_test) in [np.ndarray, pd.DataFrame]
        ) and (
            self.x_train.shape[0] == self.y_train.shape[0] and
            self.x_test.shape[0] == self.y_test.shape[0] and
            self.x_train.shape[1] == self.x_test.shape[1]
        )

    def fit(self) -> None:
        """
        Fits the learner
        :return: None
        """
        self._check_data()
        self.learner.fit(self.x_train, self.y_train)

    def predict(self, x_test):
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return self.learner.predict(x_test)
