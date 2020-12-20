from raise_utils.learners import Autoencoder
from raise_utils.data import Data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np


def test_autoencoder():
    data = Data(*train_test_split(*load_iris(return_X_y=True)))
    learner = Autoencoder(n_layers=1, n_units=[5], n_out=3, n_epochs=300)
    learner.set_data(*data)
    learner.fit()

    assert True


def test_can_encode():
    data = Data(*train_test_split(*load_iris(return_X_y=True)))
    learner = Autoencoder(n_layers=1, n_units=[5], n_out=3, n_epochs=300)
    learner.set_data(*data)
    learner.fit()

    encoded = learner.encode(data.x_test)
    assert True


def test_can_decode():
    data = Data(*train_test_split(*load_iris(return_X_y=True)))
    learner = Autoencoder(n_layers=1, n_units=[5], n_out=3, n_epochs=300)
    learner.set_data(*data)
    learner.fit()

    print(np.linalg.norm(learner.decode(
        learner.encode(data.x_test)
    ) - data.x_test))
    assert True
