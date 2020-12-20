from raise_utils.learners import MulticlassDL
from raise_utils.data import Data
from raise_utils.metrics import ClassificationMetrics
from raise_utils.hooks import Hook

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def test_dl_works():
    data = Data(*train_test_split(*load_iris(return_X_y=True)))
    data.y_train = to_categorical(data.y_train)
    data.y_test = to_categorical(data.y_test)
    learner = MulticlassDL(n_layers=2, n_units=5)
    learner.set_data(*data)
    learner.fit()

    assert True


def test_dl_can_predict():
    data = Data(*train_test_split(*load_iris(return_X_y=True)))
    data.y_train = to_categorical(data.y_train)
    data.y_test = to_categorical(data.y_test)
    learner = MulticlassDL(n_layers=2, n_units=5)
    learner.set_data(*data)
    learner.fit()

    preds = learner.predict(data.x_test)
    metrics = ClassificationMetrics(data.y_test, preds)
    metrics.add_metric('accuracy')
    metrics.get_metrics()

    assert True


def test_hooks_work():
    def pre_hook_fn(model):
        print(model.model.layers)

    def post_hook_fn(model):
        print(model.model.metrics_names)

    data = Data(*train_test_split(*load_iris(return_X_y=True)))
    data.y_train = to_categorical(data.y_train)
    data.y_test = to_categorical(data.y_test)
    learner = MulticlassDL(wfo=True, n_layers=2, n_units=5, hooks={'pre_train': [Hook(
        'pre', pre_hook_fn)], 'post_train': [Hook('post', post_hook_fn)]})
    learner.set_data(*data)
    learner.fit()

    assert True
