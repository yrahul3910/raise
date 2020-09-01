from sklearn.feature_extraction.text import HashingVectorizer

from raise_utils.data import Data
import random


class Hashing:
    def __init__(self, random=True):
        self.random = random
        self.transformer = None

    def fit_transform(self, data: Data):
        if self.random:
            a = random.choice([1000, 2000, 4000, 6000, 8000, 10000])
            b = random.choice(['l1', 'l2', None])
            vect = HashingVectorizer(n_features=a, norm=b)
        else:
            vect = HashingVectorizer()

        vect.fit_transform(data.x_train)
        self.transformer = vect

    def transform(self, x_test):
        if self.transformer is None:
            raise AssertionError("fit_transform must be called before transform.")

        return self.transformer.transform(x_test)
