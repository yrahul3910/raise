from raise_utils.data import DataLoader
from raise_utils.learners import NaiveBayes


def test_nb_works():
    clf = NaiveBayes()
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here
