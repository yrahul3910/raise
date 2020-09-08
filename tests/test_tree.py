from raise_utils.data import DataLoader
from raise_utils.learners import DecisionTree


def test_dt_works():
    clf = DecisionTree()
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here


def test_biased_tree():
    clf = DecisionTree(weighted=True)
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here
