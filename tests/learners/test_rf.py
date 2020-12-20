from raise_utils.data import DataLoader
from raise_utils.learners import RandomForest


def test_rf_works():
    clf = RandomForest()
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True


def test_weighted_rf():
    clf = RandomForest(weighted=True)
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True
