from raise_utils.data import DataLoader
from raise_utils.learners import SVM, BiasedSVM


def test_svm_works():
    clf = SVM()
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here


def test_biased_svm():
    clf = BiasedSVM()
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    clf.set_data(*data)
    clf.fit()

    assert True  # Check if we reached here
