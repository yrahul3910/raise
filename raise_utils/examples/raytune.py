"""
An example demonstrating the flexibility of the RAISE package. In this
example, we tune the hyperparameters of a deep learner with the RayTune
hyperparameter optimizer. You can install the RayTune package by running:
    pip3 install 'ray[tune]'
"""

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from raise_utils.data import DataLoader
from raise_utils.learners.feedforward_torch import FeedforwardDL
from raise_utils.metrics import ClassificationMetrics


def run(config):
    """
    This is the function RayTune optimizes. The config passed is the
    current config it is testing. This function can also take in a
    checkpoint_dir argument.
    """
    learner = FeedforwardDL(
        weighted=config['weighted'], n_layers=config['n_layers'], n_units=config['n_units'])
    learner.set_data(*data)
    learner.fit()
    preds = learner.predict(data.x_test)
    metrics = ClassificationMetrics(data.y_test, preds)
    metrics.add_metrics(['pd', 'pf'])
    result = metrics.get_metrics()
    pd = result[0]
    pf = result[1]

    """
    We want to monitor these performance measures. The keywords
    are all user-defined.
    """
    tune.report(score=pd - pf, pd=pd, pf=pf)


data = DataLoader.from_file('./promise/camel-1.2.csv')
data.y_train = (data.y_train > 0).astype('int')
data.y_test = (data.y_test > 0).astype('int')

"""This configuration object tells RayTune the search space."""
config = {
    'n_units': tune.choice(range(5, 20)),
    'n_layers': tune.choice(range(2, 5)),
    'weighted': tune.uniform(0.1, 1.)
}

scheduler = ASHAScheduler(
    metric='score',
    mode='max',
    max_t=30,
    grace_period=1,
    reduction_factor=2)

"""Report the score (pd - pf) in the terminal."""
reporter = CLIReporter(
    metric_columns=['score'])

result = tune.run(
    run,
    config=config,
    scheduler=scheduler,
    num_samples=30,
    progress_reporter=reporter
)

best_trial = result.get_best_trial("score", "max", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial score = {}".format(best_trial.last_result["score"]))
