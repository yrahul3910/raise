import warnings

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np
from raise_utils.learners.learner import Learner


# From https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
class TextDeepLearner(Learner):
    def __init__(self, epochs=10, max_words=1000, max_len=40, embedding=5, token_filters=' ',
                 n_layers=1, *args, **kwargs):
        """
        Initializes the text learner.

        :param epochs: Number of epochs to train for
        :param max_words: Maximum number of top words to consider
        :param max_len: Maximum length of sequences
        :param embedding: Embedding dimensionality
        :param token_filters: Tokens to tokenize by
        :param n_layers: Number of LSTM layers
        :param args: Args passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(TextDeepLearner, self).__init__(*args, **kwargs)
        self.epochs = epochs
        self.max_words = max_words
        self.token_filters = token_filters
        self.max_len = max_len
        self.embed_dim = embedding
        self.n_layers = n_layers

        self.random_map = {
            "max_words": (500, 5000),
            "max_len": (10, 50),
            "n_layers": (1, 4)
        }
        self.learner = self
        self._instantiate_random_vals()

    def set_data(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.y_train[self.y_train != 0] = 1
        self.y_test[self.y_test != 0] = 1

        tokenizer = Tokenizer(num_words=self.max_words,
                              filters=self.token_filters, lower=True)
        tokenizer.fit_on_texts(self.x_train)
        self.x_train = tokenizer.texts_to_sequences(self.x_train)
        self.x_test = tokenizer.texts_to_sequences(self.x_test)

        self.x_train = pad_sequences(self.x_train, maxlen=self.max_len)
        self.x_test = pad_sequences(self.x_test, maxlen=self.max_len)

    def fit(self):
        self._check_data()
        model = Sequential()
        model.add(Embedding(self.max_words, self.embed_dim,
                            input_length=self.x_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        for _ in range(self.n_layers):
            model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')

        self.learner = model

        if self.hooks is not None:
            if self.hooks.get('pre_train', None):
                for hook in self.hooks['pre_train']:
                    hook.call(self)

        model.fit(self.x_train, self.y_train,
                  batch_size=64, epochs=self.epochs)

        if self.hooks is not None:
            if self.hooks.get('post_train', None):
                for hook in self.hooks['post_train']:
                    hook.call(model)

    def predict_on_test(self) -> np.ndarray:
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return self.learner.predict_classes(self.x_test)

    def predict(self, x_test):
        """
        Overrides parent method, ignoring argument passed.

        :param x_test: Ignored.
        :return: Array of preds.
        """
        warnings.warn("predict() should not be used with TextDeepLearner. Instead, use predict_on_test" +
                      ". The argument is ignored")
        return self.predict_on_test()
