from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import backend as K
import numpy as np
import random
from raise_utils.learners.learner import Learner
from raise_utils.transform.wfo import fuzz_data


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    Variables:
    weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        return K.mean(
            K.binary_crossentropy(y_true, y_pred) * weights)

    return loss


class FeedforwardDL(Learner):
    """
    A standard feed-forward neural network.
    """

    def __init__(self, weighted=False, wfo=False, optimizer='adam', n_layers=3, n_units=19,
                 activation='relu', n_epochs=10, verbose=1, *args, **kwargs):
        """
        Initializes the deep learner.
        :param weighted: Whether to use a weighted loss function
        :param wfo: Whether to use weighted fuzzy oversampling
        :param optimizer: Choice of optimizer. Must be recognized by Keras.
        :param n_layers: Number of layers
        :param n_units: Number of units per layer
        :param activation: Activation to use
        :param n_epochs: Number of epochs
        :param verbose: Whether training should be verbose
        :param args: Args passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(FeedforwardDL, self).__init__(*args, **kwargs)

        self.activation = activation
        self.weighted = weighted
        self.wfo = wfo
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.loss = 'binary_crossentropy'

        self.learner = self
        self.model = Sequential()
        self._instantiate_random_vals()

    def fit(self):
        self._check_data()
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train).squeeze()
        self.y_test = np.array(self.y_test).squeeze()

        if self.weighted:
            frac = sum(self.y_train) * 1. / len(self.y_train)

            if isinstance(self.weighted, int):
                self.weighted = 1.
            self.loss = weighted_categorical_crossentropy(weights=(1., self.weighted / frac))

        if self.wfo:
            self.x_train, self.y_train = fuzz_data(self.x_train, self.y_train)

        for _ in range(self.n_layers):
            self.model.add(Dense(self.n_units, activation=self.activation))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        self.model.fit(np.array(self.x_train), np.array(self.y_train), batch_size=512, epochs=self.n_epochs,
                       validation_split=0.2, verbose=self.verbose)

    def predict(self, x_test) -> np.ndarray:
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return self.model.predict_classes(x_test)
