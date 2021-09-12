from raise_utils.data import Data
from raise_utils.transforms.text.hashing import Hashing
from raise_utils.transforms.text.lda import LDA
from raise_utils.transforms.text.tf import Tf
from raise_utils.transforms.text.tfidf import TfIdf

transforms = {
    "tf": Tf,
    "tfidf": TfIdf,
    "hashing": Hashing,
    "lda": LDA
}


class TextTransform:
    """Implements transforms for text data."""

    def __init__(self, name, random=True):
        """
        Initializes the transformer.

        :param name: Name of the transform to apply
        :param random: Random parameters
        """
        if name not in transforms.keys():
            raise ValueError("Invalid name.")

        self.name = name
        self.random = random
        self.transformer = None

    def fit_transform(self, data: Data) -> None:
        """
        Fits to the training data and transforms it in-place.

        :param data: A Data object.
        :return: None
        """
        self.transformer = transforms[self.name](random=self.random)
        self.transformer.fit_transform(data)

    def transform(self, x_test):
        return self.transformer.transform(x_test)
