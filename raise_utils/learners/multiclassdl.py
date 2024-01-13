from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from raise_utils.learners.learner import Learner
from raise_utils.transforms.wfo import fuzz_data
from imblearn.over_sampling import SMOTE


class MulticlassDL(Learner):
    """
    A standard feed-forward neural network.
    """

    def __init__(self, wfo=False, bs=128, n_classes=3, optimizer='adam', n_layers=3, n_units=19,
                 activation='relu', n_epochs=10, verbose=1, *args, **kwargs):
        """
        Initializes the deep learner.
        :param bs: Batch size
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
        super().__init__(*args, **kwargs)

        self.activation = activation
        self.wfo = wfo
        self.bs = bs
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.verbose = verbose
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.loss = 'categorical_crossentropy'

        self.learner = self
        self.model = Sequential()

        self.random_map = {
            'n_layers': (2, 6),
            'n_units': (3, 20),
            'activation': ['relu', 'selu']
        }
        self._instantiate_random_vals()

    def set_data(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Sets the learner data, in the order (x_train, y_train, x_test, y_test). Please pass in
        one-hot encoded values for the target. This can be done as follows:

        from tf.keras.utils import to_categorical
        y = to_categorical(y, num_classes=3)
        """
        self.x_train = x_train
        self.y_train = np.argmax(y_train, axis=-1)
        self.x_test = x_test
        self.y_test = y_test

    def fit(self):
        self._check_data()
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train).squeeze()
        self.y_test = np.array(self.y_test).squeeze()

        if self.wfo:
            self.x_train, self.y_train = fuzz_data(self.x_train, self.y_train)
            sm = SMOTE()
            self.x_train, self.y_train = sm.fit_resample(
                self.x_train, self.y_train)

        for _ in range(self.n_layers):
            self.model.add(Dense(self.n_units, activation=self.activation))

        self.model.add(Dense(self.n_classes, activation='softmax'))
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        # Last check for shapes
        if len(self.y_train.shape) == 1:
            self.y_train = to_categorical(self.y_train, num_classes=self.n_classes)

        self.model.fit(np.array(self.x_train), np.array(self.y_train), batch_size=self.bs, epochs=self.n_epochs,
                       validation_split=0.2, verbose=self.verbose, callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-3)
        ])

        if self.hooks is not None:
            if self.hooks.get('post_train', None):
                for hook in self.hooks['post_train']:
                    hook.call(self.learner)

    def predict(self, x_test):
        """
        Makes multi-class predictions, returning the classes.
        :param x_test: Test data
        :return: np.ndarray
        """
        preds = np.argmax(self.model.predict(x_test), axis=-1)
        return to_categorical(preds, num_classes=self.n_classes)
