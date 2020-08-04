from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KernelCenterer
from imblearn.over_sampling import SMOTE
from transform.cfs import CFS
import numpy as np

from data.data import Data

transformers = {
    "normalize": Normalizer,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
    "kernel": KernelCenterer,
    "standardize": StandardScaler,
    "smote": SMOTE,
    "cfs": CFS
}


class Transform:
    """An encapsulation for data transforms."""
    def __init__(self, name: str, random=False):
        """
        Initializes the Transform object.
        :param name: str. If invalid, raises a ValueError.
        :param random: bool.
        """
        self.name = name

        if name not in transformers.keys():
            raise ValueError("Invalid transform name.")

        if random:
            if name == "robust":
                start = np.random.randint(0, 50)
                end = np.random.randint(start + 1, 100)
                self.transformer = RobustScaler(quantile_range=(start, end))
            elif name == "normalize":
                norm = np.random.choice(['l1', 'l2', 'max'])
                self.transformer = Normalizer(norm=norm)
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
        if self.name != "smote":
            self.transformer.fit_transform(data.x_train, data.y_train)
            self.transformer.transform(data.x_test)
        else:
            data.x_train, data.y_train = self.transformer.fit_sample(data.x_train, data.y_train)
            data.x_test = self.transformer.sample(data.x_test)
