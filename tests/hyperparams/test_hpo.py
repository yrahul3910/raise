from copy import deepcopy
from raise_utils.data import DataLoader, Data
from raise_utils.hooks import Hook
from raise_utils.learners import FeedforwardDL
from raise_utils.transforms import Transform
from raise_utils.hyperparams import HPO, MetricObjective


options = {
    'n_units': (2, 10),
    'n_layers': (2, 6),
    'transform': ['normalize', 'standardize', 'minmax', 'maxabs', 'robust'],
    'smote': [False, True],
    'wfo': [False, True]
}


def _binarize(x, y): y[y > 1] = 1


def get_model(data: Data, config: dict) -> FeedforwardDL:
    '''
    Runs one experiment, given a Data instance.

    :param {Data} data - The dataset to run on, NOT preprocessed.
    :param {str} name - The name of the experiment.
    :param {dict} config - The config to use. Must be one in the format used in `process_configs`.
    '''
    transform = Transform(config['transform'])
    transform.apply(data)

    model = FeedforwardDL(weighted=True, wfo=config['wfo'], smote=config['smote'], n_layers=config['n_layers'],
                          n_units=config['n_units'], n_epochs=100, verbose=0)
    model.set_data(*data)

    return model


data_orig = DataLoader.from_file(
    '../promise/log4j-1.1.csv', hooks=[Hook('hook', _binarize)])


def test_hyperopt():
    data = deepcopy(data_orig)
    load_model = lambda config: get_model(data, config)
    obj = MetricObjective(['pd', 'pf', 'prec', 'auc'], data, load_model)
    hpo = HPO(
        objective=obj,
        space=options,
        algorithm='hyperopt',
        max_evals=10
    )
    config = hpo.run()

    assert isinstance(config, dict)


def test_bohb():
    obj = MetricObjective(['pd', 'pf', 'prec', 'auc'], deepcopy(data_orig), get_model)
    hpo = HPO(
        objective=obj,
        space=options,
        algorithm='bohb',
        max_evals=10
    )
    config = hpo.run()

    assert isinstance(config, dict)
