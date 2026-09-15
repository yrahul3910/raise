from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES

import numpy as np
import pytest

import raise_utils as ru
from raise_utils.learners import FeedforwardDL


def test_version():
    assert ru.__version__ != ""


def test_cython_extension():
    module = import_module("raise_utils.transforms.remove_labels")

    assert module.__file__.endswith(tuple(EXTENSION_SUFFIXES))
    assert module.Smooth.__module__ == module.__name__
    assert module.remove_labels.__module__ == module.__name__


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_cython_transform_with_numpy_integers(dtype):
    module = import_module("raise_utils.transforms.remove_labels")
    x_train = np.arange(200, dtype=np.float32).reshape(100, 2)
    y_train = np.arange(100, dtype=dtype) % 2

    x_result, y_result = module.Smooth().fit_transform(x_train, y_train)

    assert x_result.shape == x_train.shape
    assert y_result.shape == y_train.shape


def test_torch_training_and_prediction():
    keras = import_module("keras")
    assert keras.config.backend() == "torch"

    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_train = np.array([0, 1, 1, 0])
    learner = FeedforwardDL(n_layers=1, n_units=4, n_epochs=1, bs=4, verbose=0)
    learner.set_data(x_train, y_train, x_train, y_train)
    learner.fit()

    predictions = learner.predict(x_train)
    assert predictions.shape == (len(x_train), 1)
    assert np.isin(predictions, [0, 1]).all()
    assert np.isfinite(learner.model.history.history["loss"]).all()
