from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KernelCenterer
import numpy as np

from data.data import Data

transformers = {
    "normalize": Normalizer,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
    "kernel": KernelCenterer,
    "standardize": StandardScaler
}


class Transform:
    """
    An encapsulation for data transforms.
    """
    def __init__(self, name: str, random=False):
        """
        Initializes the Transform object.
        :param name: str. If invalid, raises a ValueError.
        :param random: bool.
        """
        if name not in transformers.keys():
            raise ValueError("Invalid transform name.")

        if random:
            if name == "robust":
                start = np.random.randint(0, 50)
                end = np.random.randint(start + 1, 100)
                self.transformer = RobustScaler(quantile_range=(start, end))
            elif name == "normalize":
                abs = np.random.choice(['l1', 'l2', 'max'])
                self.transformer = Normalizer(norm=abs)
            else:
                self.transformer = transformers[name]()
        else:
            self.transformer = transformers[name]()

    def apply(self, data: Data) -> None:
        """
        Applies transform in-place to a data object.
        :param data: Data object
        :return: None
        """
        self.transformer.fit_transform(data.x_train, data.y_train)
        self.transformer.transform(data.x_test)
