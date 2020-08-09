from _ctypes import Union

import pandas as pd
import numpy as np


class Learner:

    """The base Learner class."""
    def __init__(self, name: str = "rf", random: Union(bool, dict) = False):
        """
        Initializes a Learner object

        :param name: The name of the learner. Must be a recognized name.
        :param random: Whether to initialize the hyperparameters randomly.
        """
        self.random = random
        self.learner = None
        self.__name__ = name
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def set_data(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
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
        self.y_train = y_train.apply(lambda x: 0 if x == 0 else 1)
        self.y_test = y_test.apply(lambda x: 0 if x == 0 else 1)

    def _check_data(self) -> None:
        """
        Ensures data is set

        :return: None
        """
        if (
            self.x_train is None or
            self.y_train is None or
            self.x_test is None or
            self.y_test is None
        ) or (
            self.x_train.shape[0] != self.y_train.shape[0] or
            self.x_test.shape[0] != self.y_test.shape[0] or
            self.x_train.shape[1] != self.x_test.shape[1]
        ):
            raise AssertionError("Train/test data have issues.")

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
