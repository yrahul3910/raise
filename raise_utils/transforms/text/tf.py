from sklearn.feature_extraction.text import CountVectorizer
from raise_utils.data import Data
import random


class Tf:
    def __init__(self, random=True):
        self.random = random
        self.transformer = None

    def fit_transform(self, data: Data):
        if self.random:
            a, b = random.randint(100, 1000), random.randint(1, 10)
            vect = CountVectorizer(max_df=a, min_df=b)
        else:
            vect = CountVectorizer()

        data.x_train = vect.fit_transform(data.x_train)
        self.transformer = vect

    def transform(self, x_test):
        if self.transformer is None:
            raise AssertionError("fit_transform must be called before transform")

        return self.transformer.transform(x_test)
