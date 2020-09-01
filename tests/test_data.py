from raise_utils.data import DataLoader
import pandas as pd


def test_load_single_file():
    data = DataLoader.from_file("../promise/camel-1.2.csv", "bug")
    assert isinstance(data.x_train, pd.DataFrame)
    assert isinstance(data.x_test, pd.DataFrame)
    assert isinstance(data.y_train, pd.Series)
    assert isinstance(data.y_test, pd.Series)

    assert data.x_train.shape[1] == data.x_test.shape[1]
    assert data.x_train.shape[0] == data.y_train.shape[0]
    assert data.x_test.shape[0] == data.y_test.shape[0]


def test_load_multiple_files():
    data = DataLoader.from_files("../promise/", ["camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"])
    assert isinstance(data.x_train, pd.DataFrame)
    assert isinstance(data.x_test, pd.DataFrame)
    assert isinstance(data.y_train, pd.Series)
    assert isinstance(data.y_test, pd.Series)

    assert data.x_train.shape[1] == data.x_test.shape[1]
    assert data.x_train.shape[0] == data.y_train.shape[0]
    assert data.x_test.shape[0] == data.y_test.shape[0]


def test_add():
    data1 = DataLoader.from_file("../promise/camel-1.2.csv")
    data2 = DataLoader.from_file("../promise/camel-1.4.csv")

    assert len(data1 + data2) == len(data1) + len(data2)


def test_popt_data():
    data = DataLoader.from_file("../promise/camel-1.2.csv")
    assert data.get_popt_data(data.y_train).shape[1] == data.x_train.shape[1] + 2
