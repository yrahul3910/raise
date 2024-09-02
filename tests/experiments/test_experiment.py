from raise_utils.data import DataLoader
from raise_utils.experiments import Experiment
from raise_utils.learners import NaiveBayes


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
