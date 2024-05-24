from raise_utils.interpret import DODGEInterpreter, ResultsInterpreter
from raise_utils.experiments import Experiment
from raise_utils.learners import NaiveBayes
from raise_utils.hyperparams import DODGE
from raise_utils.data import DataLoader
from raise_utils.hooks import Hook
import numpy as np
import pytest


def test_empty_dodge_interpreter():
    interp = DODGEInterpreter()
    results = interp.interpret()

    assert isinstance(results, dict)
    assert len(results.keys()) == 0


def test_dodge_interpreter():
    interp = DODGEInterpreter(files=['./interpret/test.txt'], max_by=0)
    results = interp.interpret()

    assert isinstance(results, dict)
    assert 'test.txt' in results.keys()
    assert np.median(results['test.txt'][0]) - 0.892 < 1e-3


def test_dodge_interpreter_raises():
    interp = DODGEInterpreter(
        files=['./interpret/test.txt'], metrics=['accuracy'], max_by=0)

    with pytest.raises(ValueError):
        _ = interp.interpret()


def test_maxby_func():
    interp = DODGEInterpreter(
        files=['./interpret/test.txt'], max_by=lambda p: p[0]-p[1])
    results = interp.interpret()

    assert isinstance(results, dict)
    assert 'test.txt' in results.keys()


def test_results_interpreter():
    # Set up Experiment config
    exp_config = {
        "runs": 5,
        "metrics": ["accuracy"],
        "random": True,
        "learners": [NaiveBayes()],
        "log_path": "./",
        "data": [DataLoader.from_file("../promise/ant.csv")],
        "name": "test_experiment"
    }

    exp = Experiment(exp_config)
    exp.run()
    res_interp = ResultsInterpreter(['./test_experiment'])
    res_interp.interpret()

    res_interp.get_medians()
    assert True


def test_results_interpreter_with_dodge():
    # Set up Experiment config
    exp_config = {
        "runs": 5,
        "metrics": ["accuracy"],
        "random": True,
        "learners": [NaiveBayes()],
        "log_path": "./",
        "data": [DataLoader.from_file("../promise/ant.csv")],
        "name": "test_experiment"
    }

    exp = Experiment(exp_config)
    exp.run()

    # Set up DODGE config
    def _binarize(x, y): y[y > 1] = 1
    data = DataLoader.from_file(
        "../promise/ant.csv", hooks=[Hook("hook", _binarize)])

    dodge_config = {
        'n_runs': 5,
        'transforms': ['normalize', 'standardize', 'robust', 'maxabs', 'minmax'] * 30,
        'metrics': ['d2h', 'accuracy', 'pd', 'prec', 'pf'],
        'random': True,
        'learners': [NaiveBayes()] * 20,
        'log_path': './',
        'data': [data],
        'name': "test_dodge"
    }

    dodge = DODGE(dodge_config)
    dodge.optimize()

    # Set up interpreter
    def _merge(r, d): return True
    dodge_interp = DODGEInterpreter(
        ['./test_dodge.txt'], metrics=['d2h', 'accuracy', 'pd', 'prec', 'pf'])
    res_interp = ResultsInterpreter(
        ['./test_experiment']).with_dodge(dodge_interp, merge_method=_merge)

    # Also test without explicit merge given
    res_interp2 = ResultsInterpreter(
        ['./test_experiment']).with_dodge(dodge_interp)

    res_interp.interpret()
    res_interp2.interpret()

    assert True
