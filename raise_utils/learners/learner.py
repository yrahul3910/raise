import random
import pandas as pd

from raise_utils import utils
from raise_utils.data import Data

import gc


class Learner:

    """The base Learner class."""

    def __init__(self, name: str = "rf", random=False, hooks: dict = None):
        """
        Initializes a Learner object

        :param name: The name of the learner. Must be a recognized name.
        :param random: Whether to initialize the hyperparameters randomly.
        :param hooks: An object with keys "pre_train", "post_train" that correspond to pre-train and post-train hooks.
        """
        self.random = random
        self.learner = None
        self.name = name
        self.__name__ = name
        self.random_map = {}
        self.hooks = hooks
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def __del__(self):
        del self.x_train, self.x_test, self.y_train, self.y_test
        gc.collect()

    def __str__(self):
        attrs = filter(lambda x: not x.startswith(
            "_") and not callable(getattr(self, x)), dir(self))
        attr_dic = {k: getattr(self, k) for k in attrs}
        return str(attr_dic)

    def _get_random_val(self, key):
        """
        Used to fetch random hyperparameter values

        :param key: Key to search in random_map
        :return: A random value.
        """
        if not hasattr(self.learner, key):
            raise ValueError("Learner does not have key " + key)

        if isinstance(self.random_map[key], tuple):
            if isinstance(self.random_map[key][0], int):
                return random.randint(*self.random_map[key])
            return random.random() * (self.random_map[key][1] - self.random_map[key][0]) + self.random_map[key][0]
        elif isinstance(self.random_map[key], list):
            return random.choice(self.random_map[key])

    def _instantiate_random_vals(self):
        if isinstance(self.random, bool) and self.random:
            for key in self.random_map.keys():
                setattr(self.learner, key, self._get_random_val(key))
        elif isinstance(self.random, dict):
            self.random_map = self.random
            for key in self.random.keys():
                setattr(self.learner, key, self._get_random_val(key))

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

        if len(y_train.shape) == 1:
            self.y_train = pd.Series(y_train).apply(lambda x: 0 if x == 0 else 1)
            self.y_test = pd.Series(y_test).apply(lambda x: 0 if x == 0 else 1)
        else:
            self.y_train = y_train
            self.y_test = y_test

    def _check_data(self) -> None:
        """
        Ensures data is set

        :return: None
        """
        utils._check_data(Data(self.x_train, self.x_test,
                               self.y_train, self.y_test))

    def fit(self) -> None:
        """
        Fits the learner

        :return: None
        """
        self._check_data()

        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        self.learner.fit(self.x_train, self.y_train)

        if self.hooks is not None:
            if self.hooks.get('post_train', None):
                for hook in self.hooks['post_train']:
                    hook.call(self.learner)

    def predict(self, x_test):
        """
        Makes predictions

        :param x_test: Test data
        :return: np.ndarray
        """
        return self.learner.predict(x_test)
