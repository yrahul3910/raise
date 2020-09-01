from data import DataLoader
from learners import FeedforwardDL
import pytest
import numpy as np


def test_feedforward_raises_exception():
    clf = FeedforwardDL(weighted=True, wfo=True, n_layers=4, n_epochs=20)
    with pytest.raises(AssertionError):
        clf.fit()


def test_feedforward_fit():
    data = DataLoader.from_file("../promise/camel-1.2.csv", "bug")
    clf = FeedforwardDL(weighted=True, wfo=True, n_layers=4, n_units=data.x_train.shape[1], n_epochs=20)
    clf.set_data(*data)
    clf.fit()
    assert clf.predict(data.x_test).shape[0] > 0


def members(x):
    return np.array([v for k, v in x.__dict__.items() if not callable(getattr(x, k)) and not k.startswith("__")])


def test_learner_randomness():
    clf1 = FeedforwardDL(random=True)
    clf2 = FeedforwardDL(random=True)

    different = any(members(clf1) != members(clf2))
    print(members(clf1))
    print(members(clf2))
    assert different


def test_wfo_works():
    data = DataLoader.from_file("../promise/camel-1.2.csv", "bug")
    initial_size = data.x_train.shape[0]
    clf = FeedforwardDL(random=False, wfo=True)
    clf.set_data(*data)
    clf.fit()

    final_size = clf.x_train.shape[0]
    assert final_size > initial_size
