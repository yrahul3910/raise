from raise_utils.hyperparams import DODGE
from raise_utils.data import DataLoader
from raise_utils.learners import RandomForest
from raise_utils.hooks import Hook


def test_dodge():
    def _binarize(x, y): y[y > 1] = 1
    data = DataLoader.from_file(
        "../promise/log4j-1.1.csv", hooks=[Hook("hook", _binarize)])

    config = {
        'n_runs': 5,
        'transforms': ['normalize', 'standardize', 'robust', 'maxabs', 'minmax'] * 30,
        'metrics': ['d2h', 'accuracy', 'pd', 'prec', 'pf'],
        'random': True,
        'learners': [],
        'log_path': '/tmp/',
        'data': [data],
        'name': "tmp"
    }

    for _ in range(20):
        config["learners"].append(RandomForest(random=True))

    dodge = DODGE(config)
    dodge.optimize()

    assert True
