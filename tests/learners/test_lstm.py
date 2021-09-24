from raise_utils.learners import TextDeepLearner
from raise_utils.data import TextDataLoader
import pytest


def test_lstm_works():
    data = TextDataLoader.from_file('../pits/pitsA.txt')

    lstm = TextDeepLearner(epochs=10, max_words=500, embedding=7)
    lstm.set_data(*data)
    lstm.fit()
    preds = lstm.predict_on_test()

    assert True


def test_lstm_predict_warns():
    data = TextDataLoader.from_file('../pits/pitsA.txt')
    lstm = TextDeepLearner()
    lstm.set_data(*data)
    lstm.fit()

    with pytest.warns(Warning):
        lstm.predict(data.x_test)
