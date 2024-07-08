import os
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


class Data:
    """Base class for data"""

    def __add__(self, other):
        """
        Appends a Data object

        :param other: Other Data object
        :return: Data
        """
        x_train = pd.concat((self.x_train, other.x_train), axis=0)
        x_test = pd.concat((self.x_test, other.x_test), axis=0)
        y_train = pd.concat((self.y_train, other.y_train), axis=0)
        y_test = pd.concat((self.y_test, other.y_test), axis=0)
        return Data(x_train, x_test, y_train, y_test)

    def __iter__(self):
        """
        For the splat operator.

        :return x_train, y_train, x_test, y_test
        """
        return iter([self.x_train, self.y_train, self.x_test, self.y_test])

    def __init__(self, x_train, x_test, y_train, y_test):
        """
        Initializes the Data wrapper object

        :param x_train: Train data
        :param x_test: Test data
        :param y_train: Train labels
        :param y_test: Test labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return len(self.x_train)

    def get_popt_data(self, preds):
        preds = pd.Series(np.array(preds).squeeze(), name="prediction")
        self.x_test.reset_index(inplace=True, drop=True)
        self.y_test.reset_index(drop=True, inplace=True)
        if isinstance(self.x_train, pd.DataFrame):
            return pd.concat((self.x_test, self.y_test, preds), axis=1)
        else:
            df = pd.DataFrame(self.x_test)
            df["bug"] = self.y_test
            df["prediction"] = preds
            return df


class DataLoader:
    """Data loading utilities"""

    @staticmethod
    def from_files(base_path: str, files: list, target: str | int = "bug", col_start: int = 3, col_stop: int = -2, n_classes: int = 2, hooks: list = None, **pd_kwargs) -> Data:
        """
        Builds data from a list of files, the last of which is the test set.

        :param base_path: The path to fetch from
        :param files: List of files. The last one is the test set.
        :param target: Target column
        :param col_start: Column to start reading from. 3 for PROMISE defect prediction.
        :param col_stop: Column to stop reading. -2 for PROMISE defect prediction.
        :param n_classes: Number of classes.
        :param hooks: List of hooks. These are passed the train/test DataFrames after
            filtering columns
        :return: Data object
        """
        paths = [os.path.join(base_path, file_name) for file_name in files]
        train_df = pd.concat([pd.read_csv(path, **pd_kwargs)
                              for path in paths[:-1]], ignore_index=True)
        test_df = pd.read_csv(paths[-1], **pd_kwargs)

        train_df, test_df = train_df.iloc[:,
                                          col_start:], test_df.iloc[:, col_start:]
        df = pd.concat([train_df, test_df], ignore_index=True)

        if isinstance(target, int):
            target = df.columns[target]
        elif not isinstance(target, str):
            raise ValueError("Target must be a string or an integer")

        if n_classes == 2:
            df[target] = df[target].apply(lambda x: 0 if x == 0 else 1)
        elif n_classes > 2:
            df[target] = to_categorical(
                df[target], num_classes=n_classes) 

        train_size = len(train_df)
        train_data = df.iloc[:train_size, :]
        test_data = df.iloc[train_size:, :]

        if hooks is not None:
            for hook in hooks:
                hook.call(train_data, test_data)

        X_train = train_data[train_data.columns[:col_stop]]
        y_train = train_data[target].astype("int")
        X_test = test_data[test_data.columns[:col_stop]]
        y_test = test_data[target].astype("int")

        return Data(X_train, X_test, y_train, y_test)

    @staticmethod
    def from_file(path: str, target: str | int = "bug", col_start: int = 3, col_stop: int = -2, hooks: list = None, **pd_kwargs) -> Data:
        """
        Path to file

        :param path: Path to file
        :param target: Target column
        :param col_start: Column to start reading at
        :param col_stop: Column to stop reading at
        :param hooks: List of hooks. This is passed after the columns are filtered.
            The data is passed as a DataFrame (x) and a Series (y), before splitting for
            train/test.
        :return: Data object
        """
        df = pd.read_csv(path, **pd_kwargs)

        if isinstance(target, str):
            y = df[target].astype("int")
        elif isinstance(target, int):
            y = df.iloc[:, target].astype("int")
        else:
            raise ValueError("Target must be a string or an integer")

        x = df.drop(columns=target)
        x = x.iloc[:, col_start:col_stop]

        if hooks is not None:
            for hook in hooks:
                hook.call(x, y)

        return Data(*train_test_split(x, y))


class TextDataLoader:
    """Class for loading text data."""

    @ staticmethod
    def from_file(filename, splitter=">>>"):
        """
        Reads data from a file, where data and labels are separated by splitter.

        :param filename: Path to file to read.
        :param splitter: Splitting text
        :return: Data object
        """
        dic = []
        labels = []
        with open(filename, 'r') as f:
            for doc in f.readlines():
                row = doc.lower().split(splitter)
                dic.append(row[0].strip())
                labels.append(row[1].strip())
        count = Counter(labels)
        import operator
        key = max(count.items(), key=operator.itemgetter(1))[0]
        labels = list(map(lambda x: 1 if x == key else 0, labels))
        x, y = np.array(dic), pd.Series(labels)

        return Data(*train_test_split(x, y))
