import pytest
import math
from raise_utils.metrics import ClassificationMetrics


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
                         'd2h', 'f1', 'prec', 'conf'])
    results = metrics.get_metrics()

    assert len(results) == 8
    assert results[0] == 0.5  # Accuracy
    assert results[1] == 0.5  # pf
    assert results[2] == 0.5  # recall
    assert results[3] == results[2]
    assert results[4] == 1. / math.sqrt(2) - 1. / 2
    assert results[5] - 4 / 7 < 1e-3
    assert results[6] == 2. / 3
    assert results[7] == (2, 1, 1, 2)
