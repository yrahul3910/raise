from raise_utils.experiments import Experiment
from raise_utils.learners import NaiveBayes
from raise_utils.data import DataLoader
from raise_utils.hooks import Hook


def _call(learner, data):
    assert learner is not None
    assert data is not None
    assert len(data.x_test) > 0


def test_empty_transform():
    config = {
        "runs": 5,
        "metrics": ["accuracy"],
        "random": True,
        "learners": [NaiveBayes()],
        "log_path": "/dev/null",
        "data": [DataLoader.from_file("../promise/ant.csv")],
        "name": ""
    }

    exp = Experiment(config)

    assert exp.transforms == []


def test_experiment_runs():
    config = {
        "runs": 5,
        "metrics": ["accuracy"],
        "random": True,
        "learners": [NaiveBayes()],
        "log_path": "/dev/null",
        "data": [DataLoader.from_file("../promise/ant.csv")],
        "name": ""
    }

    exp = Experiment(config)
    exp.run()

    assert True


def test_multiple_datasets():
    config = {
        "runs": 5,
        "metrics": ["accuracy"],
        "transforms": ["standardize"],
        "random": True,
        "learners": [NaiveBayes()],
        "log_path": "/dev/null",
        "data": [DataLoader.from_file("../promise/ant.csv"), DataLoader.from_file("../promise/camel.csv")],
        "name": ""
    }

    exp = Experiment(config)
    exp.run()

    assert True


def test_popt():
    def _binarize(x, y):
        y[y > 1] = 1
    data1 = DataLoader.from_file(
        "../promise/ant.csv", hooks=[Hook("hook", _binarize)], col_start=0)
    data2 = DataLoader.from_file(
        "../promise/camel.csv", hooks=[Hook("hook", _binarize)], col_start=0)

    config = {
        "runs": 5,
        "metrics": ["accuracy", "popt20"],
        "transforms": ["standardize"],
        "random": True,
        "learners": [NaiveBayes()],
        "log_path": "/dev/null",
        "post_train_hooks": [Hook("hook", _call)],
        "data": [data1, data2],
        "name": ""
    }

    exp = Experiment(config)
    exp.run()
