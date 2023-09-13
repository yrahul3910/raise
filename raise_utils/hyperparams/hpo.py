"""
Implements a general class for multiple HPO algorithms. For now, only hyperopt and BOHB are supported.
This class expects the hpo_space to be defined in the following format:
{
    "param1": [categories],
    "param2": (lower, upper),  # lower inclusive, upper exclusive
    ...
}
For generality, this class expects an objective function that is *minimized*. This class also
provides some general objective functions for convenience.
"""
from typing import Callable

from raise_utils.data import Data
from raise_utils.metrics import ClassificationMetrics
from hyperopt import hp, fmin, tpe
from bohb import BOHB
import bohb.configspace as bohb_space
import numpy as np


class MetricObjective:
    def __init__(self, metrics: list, data: Data, get_model: Callable):
        self.metrics = metrics
        self.data = data
        self.get_model = get_model

    def __call__(self, config, *args, **kwargs):
        model = self.get_model(config)
        model.set_data(*self.data)
        model.fit()

        metrics = ClassificationMetrics(self.data.y_test, model.predict(self.data.x_test))
        metrics.add_metrics(self.metrics)
        return -metrics.get_metrics()[0]


class HPO:
    ALGORITHMS = ["random", "hyperopt", "bohb"]

    def __init__(self, objective: Callable, space: dict, algorithm: str, max_evals: int = 30):
        self.objective = objective
        self.hpo_space = space
        self.max_evals = max_evals

        if algorithm not in self.ALGORITHMS:
            raise ValueError("Algorithm must be one of " + str(self.ALGORITHMS))

        self.algorithm = algorithm

    def _run_bohb(self):
        # Convert the space to bohb format
        space = []
        for key, val in self.hpo_space.items():
            if isinstance(val, list):
                space.append(bohb_space.CategoricalHyperparameter(key, val))
            elif isinstance(val, tuple):
                if isinstance(val[0], int):
                    space.append(bohb_space.IntegerUniformHyperparameter(key, *val))
                else:
                    space.append(bohb_space.UniformHyperparameter(key, *val))
            else:
                raise ValueError(f"Key {key} must be a list or tuple")

        opt = BOHB(configspace=bohb_space.ConfigurationSpace(space), evaluate=self.objective, min_budget=1, max_budget=self.max_evals)
        logs = opt.optimize()
        return logs.best["hyperparameter"].to_dict()

    def _run_random(self):
        # Run random search and return the best config
        best = None
        best_score = float("inf")
        for _ in range(self.max_evals):
            config = {}
            for key, val in self.hpo_space.items():
                if isinstance(val, list):
                    config[key] = np.random.choice(val)
                elif isinstance(val, tuple) and isinstance(val[0], int):
                    config[key] = np.random.randint(val[0], val[1])
                else:
                    config[key] = np.random.uniform(val[0], val[1])

            score = self.objective(config)
            if score < best_score:
                best_score = score
                best = config

        return best

    def _run_hyperopt(self):
        # Convert the space to hyperopt format
        space = {}
        for key, val in self.hpo_space.items():
            if isinstance(val, list):
                space[key] = hp.choice(key, val)
            elif isinstance(val, tuple):
                if isinstance(val[0], int):
                    space[key] = hp.randint(key, *val)
                else:
                    space[key] = hp.uniform(key, *val)
            else:
                raise ValueError("Space must be a list or tuple")

        best = fmin(self.objective, space, algo=tpe.suggest, max_evals=self.max_evals)
        for key, val in self.hpo_space.items():
            if isinstance(val, list):
                best[key] = val[best[key]]
        return best

    def run(self):
        if self.algorithm == "hyperopt":
            return self._run_hyperopt()
        elif self.algorithm == "bohb":
            return self._run_bohb()
        elif self.algorithm == "random":
            return self._run_random()
