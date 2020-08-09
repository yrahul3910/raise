from keras import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras.utils import to_categorical

import numpy as np
from learners.learner import Learner


class TextDeepLearner(Learner):
    def __init__(self, epochs=10, *args, **kwargs):
        super(TextDeepLearner, self).__init__(*args, **kwargs)
        self.epochs = epochs

    def set_data(self, x_train, y_train, x_test, y_test) -> None:
        super(TextDeepLearner, self).set_data(x_train, y_train, x_test, y_test)
        self.x_train = np.reshape(self.x_train, (len(self.x_train), len(self.x_train[0]), 1))
        self.x_test = np.reshape(self.x_test, (len(self.x_test), len(self.x_test[0]), 1))

        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def fit(self):
        self._check_data()
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(256))
        model.add(Dropout(0.2))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(self.x_train, self.y_train, batch_size=64, epochs=self.epochs)
        self.learner = model

    def predict(self, x_test) -> np.ndarray:
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return self.learner.predict_classes(x_test)
