from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KernelCenterer
from imblearn.over_sampling import SMOTE
from raise_utils.transform.cfs import CFS
from raise_utils.transform.inliers import OutlierRemoval
from raise_utils.transform.null import NullTransform
from raise_utils.transform.text.transform import TextTransform
from raise_utils.transform.wfo import WeightedFuzzyOversampler
from raise_utils.transform.wfo import RadiallyWeightedFuzzyOversampler
import numpy as np

from raise_utils.data.data import Data

transformers = {
    "normalize": Normalizer,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "robust": RobustScaler,
    "kernel": KernelCenterer,
    "standardize": StandardScaler,
    "smote": SMOTE,
    "cfs": CFS,
    "rwfo": RadiallyWeightedFuzzyOversampler,
    "wfo": WeightedFuzzyOversampler,
    "outlier": OutlierRemoval,
    "none": NullTransform
}

text_transforms = [
    "tf",
    "tfidf",
    "hashing",
    "lda"
]


class Transform:
    """An encapsulation for data transforms."""

    def __init__(self, name: str, random=False):
        """
        Initializes the Transform object.
        :param name: str. If invalid, raises a ValueError.
        :param random: bool.
        """
        self.name = name

        if name not in transformers.keys() and name not in text_transforms:
            raise ValueError("Invalid transform name.")

        if name in transformers.keys():
            if random:
                if name == "robust":
                    start = np.random.randint(0, 50)
                    end = np.random.randint(start + 1, 100)
                    self.transformer = RobustScaler(
                        quantile_range=(start, end))
                elif name == "normalize":
                    norm = np.random.choice(['l1', 'l2', 'max'])
                    self.transformer = Normalizer(norm=norm)
                else:
                    self.transformer = transformers[name]()
            else:
                self.transformer = transformers[name]()
        else:
            self.transformer = TextTransform(name=name, random=random)

    def apply(self, data: Data) -> None:
        """
        Applies transform in-place to a data object.
        :param data: Data object
        :return: None
        """
        if self.name in text_transforms:
            self.transformer.fit_transform(data)
            data.x_test = self.transformer.transform(data.x_test)
        else:
            if self.name != "smote":
                if self.name in ["wfo", "cfs", "rwfo"]:
                    data.x_train, data.y_train = self.transformer.fit_transform(
                        data.x_train, data.y_train)
                else:
                    data.x_train = self.transformer.fit_transform(data.x_train)

                if self.name != "wfo" and self.name != "rwfo":
                    data.x_test = self.transformer.transform(data.x_test)
            else:
                data.x_train, data.y_train = self.transformer.fit_sample(
                    data.x_train, data.y_train)
