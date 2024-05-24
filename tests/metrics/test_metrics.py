import pytest
import math
import numpy as np
from raise_utils.metrics import ClassificationMetrics
from raise_utils.metrics.dist import get_jensen_shannon, get_smape


def test_invalid_metrics_rejected():
    metrics = ClassificationMetrics([1], [1])
    with pytest.raises(ValueError):
        metrics.add_metric('fail')


def test_popt20_needs_add_data():
    metrics = ClassificationMetrics([1], [1])
    with pytest.raises(AssertionError):
        metrics.add_metric('popt20')


def test_metrics():
    metrics = ClassificationMetrics([1, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 0])
    metrics.add_metrics(['accuracy', 'pf', 'pd', 'recall',
                         'd2h', 'f1', 'prec', 'conf', 'd2h2'])
    results = metrics.get_metrics()

    assert len(results) == 9
    assert results[0] == 0.5  # Accuracy
    assert results[1] == 0.5  # pf
    assert results[2] == 0.5  # recall
    assert results[3] == results[2]
    assert results[4] == 1. / math.sqrt(2) - 1. / 2
    assert results[5] - 4 / 7 < 1e-3
    assert results[6] == 2. / 3
    assert results[7] == (2, 1, 1, 2)
    assert results[8] == 1. / math.sqrt(2) - math.sqrt(3./2) / 2


def test_js_when_equal():
    np.random.seed(0)
    p = np.random.randn(10, 10)
    q = np.random.randn(10, 10)

    js = get_jensen_shannon(p, q)
    assert js < 0.1


def test_js_when_different():
    np.random.seed(0)
    p = np.random.randn(10, 10)
    q = np.random.randn(10, 10) + 1

    js = get_jensen_shannon(p, q)
    assert js > 0.0
    assert js < 1.0


def test_smape_when_equal():
    p = np.zeros((10, 10))
    q = np.zeros((10, 10))

    smape = get_smape(p, q)
    assert smape == 0.0


def test_smape_when_different():
    p = np.ones((10, 10))
    q = np.zeros((10, 10))

    smape = get_smape(p, q)
    assert smape - 1.0 <= np.finfo(float).eps
