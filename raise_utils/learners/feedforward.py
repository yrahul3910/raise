import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import numpy as np
from raise_utils.learners.learner import Learner
from raise_utils.transform.wfo import fuzz_data
from imblearn.over_sampling import SMOTE


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

    def __init__(self, weighted=False, wfo=False, smote=None, optimizer='adam', n_layers=3, n_units=19,
                 activation='relu', n_epochs=10, verbose=1, *args, **kwargs):
        """
        Initializes the deep learner.
        :param weighted: Whether to use a weighted loss function
        :param wfo: Whether to use weighted fuzzy oversampling
        :param smote: Whether or not to use SMOTE. This can be used individually, and
        is applied after weighted fuzzy oversampling. Leaving it to None will use the
        recommended settings.
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

        if (self.wfo and smote is None) or (smote == True):
            self.smote = True
        else:
            self.smote = False

        self.learner = self
        self.model = Sequential()

        self.random_map = {
            'n_layers': (2, 6),
            'n_units': (3, 20)
        }
        self._instantiate_random_vals()

    def set_data(self, x_train, y_train, x_test, y_test):
        super().set_data(x_train, y_train, x_test, y_test)

        if tf.__version__ >= '2.0.0':
            # We are running TF 2.0, so need to type cast.
            self.y_train = self.y_train.astype('float32')

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
            self.loss = weighted_categorical_crossentropy(
                weights=(1., self.weighted / frac))

        if self.wfo:
            self.x_train, self.y_train = fuzz_data(self.x_train, self.y_train)

            if self.smote:
                sm = SMOTE()
                self.x_train, self.y_train = sm.fit_sample(
                    self.x_train, self.y_train)

        for _ in range(self.n_layers):
            self.model.add(Dense(self.n_units, activation=self.activation))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        self.model.fit(np.array(self.x_train), np.array(
            self.y_train), epochs=self.n_epochs, verbose=self.verbose)
        if self.hooks is not None:
            if self.hooks.get('post_train', None):
                for hook in self.hooks['post_train']:
                    hook.call(self.model)

    def predict(self, x_test) -> np.ndarray:
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return (self.model.predict(x_test) > 0.5).astype('int32')
