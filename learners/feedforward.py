from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np
import random
from learners.learner import Learner


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


def fuzz_data(X, y, radii=(0., .3, .03)):
    idx = np.where(y == 1)[0]
    frac = len(idx) * 1. / len(y)

    fuzzed_x = []
    fuzzed_y = []

    for row in X[idx]:
        for i, r in enumerate(np.arange(*radii)):
            for j in range(int((1. / frac) / pow(2., i))):
                fuzzed_x.append([val - r for val in row])
                fuzzed_x.append([val + r for val in row])
                fuzzed_y.append(1)
                fuzzed_y.append(1)

    return np.concatenate((X, np.array(fuzzed_x)), axis=0), np.concatenate((y, np.array(fuzzed_y)))


class FeedforwardDL(Learner):
    """
    A standard feed-forward neural network.
    """
    def __init__(self, weighted=False, wfo=False, optimizer='adam', n_layers=3, n_units=19,
                 activation='relu', n_epochs=10, *args, **kwargs):
        """
        Initializes the deep learner.
        :param weighted: Whether to use a weighted loss function
        :param wfo: Whether to use weighted fuzzy oversampling
        :param optimizer: Choice of optimizer. Must be recognized by Keras.
        :param n_layers: Number of layers
        :param n_units: Number of units per layer
        :param activation: Activation to use
        :param n_epochs: Number of epochs
        :param args: Args passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(FeedforwardDL, self).__init__(*args, **kwargs)
        self.activation = activation
        self.weighted = weighted
        self.wfo = wfo
        self.optimizer = optimizer
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.loss = 'categorical_crossentropy'

        if self.random:
            self.weighted = random.choice([1., 10., 100., False])
            self.wfo = random.choice([True, False])
            self.n_layers = random.randint(1, 5)
            self.n_units = random.randint(1, 20)

    def fit(self):
        self._check_data()

        if self.weighted:
            frac = sum(self.y_train) * 1. / len(self.y_train)

            if isinstance(self.weighted, int):
                self.weighted = 1.
            self.loss = weighted_categorical_crossentropy(weights=(1., self.weighted / frac))

        if self.wfo:
            self.x_train, self.y_train = fuzz_data(self.x_train, self.y_train)

        self.learner = Sequential([
            Dense(self.x_train.shape[1], activation=self.activation)
        ])

        for _ in range(self.n_layers - 1):
            self.learner.add(Dense(self.x_train.shape[1], activation=self.activation))

        self.learner.add(Dense(len(np.unique(self.y_train)), activation='softmax'))
        self.learner.compile(optimizer=self.optimizer, loss=self.loss)

        self.learner.fit(np.array(self.x_train), np.array(self.y_train), batch_size=128, epochs=self.n_epochs,
                         validation_split=0.2)

    def predict(self, x_test) -> np.ndarray:
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return self.learner.predict_classes(x_test)
