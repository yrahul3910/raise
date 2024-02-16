from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import keras
import numpy as np
from raise_utils.learners.learner import Learner
from raise_utils.transforms.wfo import fuzz_data
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

if keras.config.backend() == "torch":
    from torch import Tensor


class FeedforwardDL(Learner):
    """
    A standard feed-forward neural network.
    """

    def __init__(self, weighted=False, bs=128, wfo=False, smote=None, optimizer='adam', n_layers=3, n_units=19,
                 activation='relu', n_epochs=10, verbose=1, *args, **kwargs):
        """
        Initializes the deep learner.
        :param weighted: Whether to use a weighted loss function
        :param bs: Batch size
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
        self.bs = bs
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

        self.y_train = self.y_train.astype('float32')

    def fit(self):
        self._check_data()

        if keras.config.backend() == "torch" and isinstance(self.x_train, Tensor):
            self.x_train = self.x_train.detach().numpy()
            self.x_test = self.x_test.detach().numpy()
        else:
            self.x_train = np.array(self.x_train)
            self.x_test = np.array(self.x_test)
            self.y_train = np.array(self.y_train).squeeze()
            self.y_test = np.array(self.y_test).squeeze()

        if self.weighted:
            weights = class_weight.compute_class_weight('balanced',
                                                        classes=np.unique(self.y_train),
                                                        y=self.y_train)
            weights = {i: weights[i] for i in range(len(weights))}
        else:
            weights = None

        if self.wfo:
            self.x_train, self.y_train = fuzz_data(self.x_train, self.y_train)

            if self.smote:
                sm = SMOTE()

                try:
                    self.x_train, self.y_train = sm.fit_resample(
                        self.x_train, self.y_train)
                except ValueError:
                    print('[WARN] FeedforwardDL: SMOTE failed.')

        for _ in range(self.n_layers):
            self.model.add(Dense(self.n_units, activation=self.activation))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        early_stopping = EarlyStopping(monitor='loss', patience=10)
        self.model.fit(np.array(self.x_train), np.array(
            self.y_train), epochs=self.n_epochs, batch_size=self.bs,
            callbacks=[early_stopping], verbose=self.verbose, class_weight=weights)
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
