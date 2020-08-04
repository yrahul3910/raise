import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Data:
    """Base class for data"""
    def __add__(self, other):
        x_train = np.concatenate((self.x_train, other.x_train), axis=0)
        x_test = np.concatenate((self.x_test, other.x_test), axis=0)
        y_train = np.concatenate((self.y_train, other.y_train), axis=0)
        y_test = np.concatenate((self.y_test, other.y_test), axis=0)

        return Data(x_train, x_test, y_train, y_test)

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_popt_data(self):
        return np.concatenate((self.x_train, self.y_train), axis=1)


class DataLoader:
    """Data loading utilities"""
    @staticmethod
    def from_files(base_path: str, files: list, target:str = "bug", col_start:int = 3, col_stop:int = -2) -> Data:
        """
        Builds data from a list of files, the last of which is the test set.

        :param base_path: The path to fetch from
        :param files: List of files. The last one is the test set.
        :param target: Target column
        :param col_start: Column to start reading from. 3 for PROMISE defect prediction.
        :param col_stop: Column to stop reading. -2 for PROMISE defect prediction.
        :return: Data object
        """
        paths = [os.path.join(base_path, file_name) for file_name in files]
        train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
        test_df = pd.read_csv(paths[-1])

        train_df, test_df = train_df.iloc[:, col_start:], test_df.iloc[:, col_start:]
        train_size = train_df[target].count()
        df = pd.concat([train_df, test_df], ignore_index=True)
        df[target] = df[target].apply(lambda x: 0 if x == 0 else 1)

        train_data = df.iloc[:train_size, :]
        test_data = df.iloc[train_size:, :]

        X_train = train_data[train_data.columns[:col_stop]]
        y_train = train_data[target]
        X_test = test_data[test_data.columns[:col_stop]]
        y_test = test_data[target]

        return Data(X_train, X_test, y_train, y_test)

    @staticmethod
    def from_file(path:str, target) -> Data:
        """
        Path to file

        :param path: Path to file
        :param target: Target column
        :return: Data object
        """
        df = pd.read_csv(path)
        y = df[target]
        x = df.drop(columns=target)

        return Data(*train_test_split(x, y))
