from raise_utils.hyperparams import BinaryGHOST, GHOST
from raise_utils.data import DataLoader
from raise_utils.metrics import ClassificationMetrics
from raise_utils.hooks import Hook


def test_ghost():
    def _binarize(x, y): y[y > 1] = 1
    data = DataLoader.from_file(
        "../promise/log4j-1.1.csv", hooks=[Hook("hook", _binarize)])

    ghost = BinaryGHOST(n_runs=1, metrics=['d2h', 'accuracy', 'pd', 'prec', 'pf'])
    ghost.set_data(*data)
    ghost.fit()

    assert True


def test_multi_ghost():
    def _binarize(x, y): y[y > 1] = 1

    def _obj_fn(y_true, y_preds):
        cm = ClassificationMetrics(y_true, y_preds)
        cm.add_metrics(['d2h'])
        return 1. - cm.get_metrics()[0]

    data = DataLoader.from_file(
        "../promise/log4j-1.1.csv", hooks=[Hook("hook", _binarize)])

    ghost = GHOST(obj_fn=_obj_fn, metrics=['d2h'])
    ghost.set_data(*data)
    ghost.fit()

    assert True
