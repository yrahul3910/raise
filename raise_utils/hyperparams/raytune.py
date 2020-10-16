import random
from typing import Union, Callable
from functools import partial

from raise_utils.data import Data
from raise_utils.learners.learner import Learner
from raise_utils.metrics import ClassificationMetrics
from raise_utils.utils import _check_data
from raise_utils.hooks import Hook

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


class RayTune:
    """
    Uses the RayTune hyperparameter optimizer to tune a (single)
    learner.
    """

    def __init__(self, fn: Union[Callable, Learner], data: Data, metrics: ClassificationMetrics, config: dict, post_results_hook: Hook, num_samples: int = 30):
        """
        Initializes all the settings for the RayTune hyperparameter
        tuner.

        Args:
        =====
        fn: Union[Callable, Learner] - Either a function or a Learner. If a function, then it is
        assumed to be the function passed to RayTune. This function is passed the `config` object.
        If a Learner instance, the keys of `config` will be passed to the instance.
        data: Data - A Data object.
        config: dict - The config with the hyper-parameters to tune. The syntax is the same as the
        randomization feature in `raise_utils.learners`. See the docs for details.
        num_samples: int - Number of samples to test, passed to RayTune.
        metrics: ClassificationMetrics - A Metrics instance with add_metrics already called.
        post_results_hook: Hook - The hook called on each iteration after the metrics are obtained.
        It is passed the config and the metrics.
        """
        if not callable(fn) and not isinstance(fn, Learner):
            raise TypeError(
                "Invalid type for fn. Must be a Callable or Learner, got " + type(fn))

        self.fn = fn

        _check_data(data)
        self.data = data

        self.config = config
        self.num_samples = num_samples
        self.metrics = metrics
        self.post_results_hook = post_results_hook

    def _get_runner(self, fn) -> Callable:
        """
        Gets the run function needed by RayTune from the passed fn.

        Args:
        =====
        fn: Union[Callable, Learner] - The fn passed to __init__

        Returns:
        ========
        run: Callable - A function to be passed to RayTune.
        """
        if callable(fn):
            def run(config, *args):
                return fn(config)

            # Ignore any other passed args.
            return run

        def run(config, metric_names):
            learner = fn(**config)
            learner.set_data(self.data)
            learner.fit()
            preds = learner.predict(self.data.x_test)
            result = self.metrics.get_metrics()

            if self.post_results_hook is not None:
                self.post_results_hook(config, result)

            if metric_names is None:
                metric_names = list(range(len(result)))

            metrics_dict = {name: result[i]
                            for i, name in enumerate(metric_names)}
            tune.report(score=result[0], **metrics_dict)

        return partial(run, self.config)

    def optimize(self, metric_names: Union[None, list]) -> None:
        """
        Runs the RayTune optimizer.

        Args:
        =====
        metric_names: Union[None, list] - If fn is a Callable, ignored. Otherwise,
        if None, assigns numbers to each metric returned by ClassificationMetrics;
        otherwise, the metric names in the list are used when printing output from
        RayTune.
        """
        scheduler = ASHAScheduler(
            metric='score',
            mode='max',
            max_t=30,
            grace_period=1,
            reduction_factor=2)

        reporter = CLIReporter(
            metric_columns=['score'])

        result = tune.run(
            self._get_runner(self.fn),
            config=config,
            scheduler=scheduler,
            num_samples=30,
            progress_reporter=reporter
        )

        best_trial = result.get_best_trial("score", "max", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial score = {best_trial.last_result['score']}")
