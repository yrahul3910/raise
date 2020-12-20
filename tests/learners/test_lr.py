from raise_utils.data import DataLoader
from raise_utils.learners import LogisticRegressionClassifier


def test_lr_works():
    clf = LogisticRegressionClassifier()
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here


def test_biased_lr():
    clf = LogisticRegressionClassifier(weighted=True)
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here
