from raise_utils.transform import Transform
from raise_utils.data import DataLoader
import numpy as np


def test_null():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')

    len_pre = data.x_train.shape[0]
    transform = Transform('none')
    transform.apply(data)

    len_post = data.x_train.shape[0]
    assert len_pre == len_post


def test_normalize():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')

    len_pre = data.x_train.shape[0]
    transform = Transform('normalize')
    transform.apply(data)

    len_post = data.x_train.shape[0]
    assert len_pre == len_post
    assert np.linalg.norm(data.x_train[0]) - 1. < 1e-3


def test_wfo():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')

    len_pre = data.x_train.shape[0]
    transform = Transform('wfo')
    transform.apply(data)

    len_post = data.x_train.shape[0]
    assert len_pre < len_post


def test_rwfo():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')

    len_pre = data.x_train.shape[0]
    transform = Transform('rwfo')
    transform.apply(data)

    len_post = data.x_train.shape[0]
    assert len_pre < len_post


def test_cfs():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')

    len_pre = data.x_train.shape[1]
    transform = Transform('cfs')
    transform.apply(data)

    len_post = data.x_train.shape[1]
    assert len_pre > len_post
