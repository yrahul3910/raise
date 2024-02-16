from raise_utils.transforms import Transform
from raise_utils.transforms.cfs import CFS
from raise_utils.transforms.wfo import WeightedFuzzyOversampler
from raise_utils.transforms.text.tfidf import TfIdf
from raise_utils.data import DataLoader
import numpy as np
import pytest


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


def test_wfo_transform_raises_error():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')
    transform = WeightedFuzzyOversampler()

    with pytest.raises(NotImplementedError):
        transform.transform(data.x_test)


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


def test_cfs_raises():
    data = DataLoader.from_file('../promise/log4j-1.1.csv')
    transform = CFS()

    with pytest.raises(AssertionError):
        transform.transform(data.x_test)


def test_hasing_works():
    data = TextDataLoader.from_file('../pits/pitsA.txt')
    transform = Transform('hashing', random=True)
    transform.apply(data)

    assert True


def test_tfidf_works():
    data = TextDataLoader.from_file('../pits/pitsA.txt')
    transform = Transform('tfidf', random=True)
    transform.apply(data)

    assert True


def test_tf_works():
    data = TextDataLoader.from_file('../pits/pitsA.txt')
    transform = Transform('tf', random=True)
    transform.apply(data)

    assert True


def test_lda_works():
    data = TextDataLoader.from_file('../pits/pitsA.txt')
    transform = Transform('tf')
    transform.apply(data)
    transform = Transform('lda', random=True)
    transform.apply(data)

    assert True


def test_text_transformers_raise_err():
    tfidf = TfIdf(random=True)

    with pytest.raises(AssertionError):
        tfidf.transform([])
