import ast
from raise_utils.interpret.sk import Rx
from statistics import median
from typing import Union, Callable
import numpy as np


class DODGEInterpreter:
    """Interprets the results of DODGE-generated files"""

    def __init__(self, files=None, max_by: Union[None, int, Callable[..., str]] = None,
                 exclude_cols=None, metrics=None, n_iters=30, settings=True) -> None:
        """
        Initializes the interpreter.

        :param files - A list of files to be interpreted.
        :param max_by - Either a None, int, or Callable. If None, defaults to
                        maximizing the first entry, the metric maximized by DODGE.
                        If int, maximizes by the index specified.
                        If callable, maximizes by the function passed.
        :param exclude_cols - List of column indices to exclude
        :param metrics - List of metrics passed to DODGE. If excluding columns,
                        do NOT include these in this list.
        :param n_iters - The n_iter setting passed to DODGE.
        :param settings - Provides compatibility for older versions of DODGE, that
                        did not print settings. For those versions, set to False.
        :return DODGEInterpreter object
        """
        if files is None:
            files = []
        if exclude_cols is None:
            exclude_cols = []
        if metrics is None:
            metrics = []
        self.files = files
        if max_by is None:
            self.max_by = 0
        else:
            self.max_by = max_by
        self.exclude_cols = exclude_cols
        self.metrics = metrics
        self.n_iters = n_iters
        self.settings = settings

    def interpret(self) -> dict:
        DODGE_ITER = self.n_iters
        medians = {}

        for file in self.files:
            with open(file, 'r') as f:
                lines = f.readlines()

            if self.settings:
                settings = [line.split(':')[1]
                            for line in lines if line.startswith('setting')]

            lines = [ast.literal_eval(line.split(':')[1])
                     for line in lines if line.startswith('iter')]

            n_runs = int(len(lines) // DODGE_ITER)
            n_metrics = len(lines[0]) - len(self.exclude_cols)

            if len(self.metrics) == 0:
                self.metrics = list(range(n_metrics))
            elif len(self.metrics) != n_metrics:
                raise ValueError("Passed list of metrics has size", len(self.metrics),
                                 "but file metrics (excluding exclude_cols) has size",
                                 n_metrics)

            lines = np.array(lines)
            lines = np.delete(lines, self.exclude_cols, -1)

            if self.settings:
                settings = np.array(settings)

            assert lines.shape == (n_runs * DODGE_ITER, n_metrics)

            run_splits = lines.reshape(
                (n_runs, DODGE_ITER, n_metrics))

            if self.settings:
                settings = settings.reshape((n_runs, DODGE_ITER))

            if isinstance(self.max_by, int):
                mapped_vals = np.apply_along_axis(
                    lambda x: x[self.max_by], axis=-1, arr=run_splits)
            elif callable(self.max_by):
                mapped_vals = np.apply_along_axis(
                    self.max_by, axis=-1, arr=run_splits)

            assert mapped_vals.shape == (n_runs, DODGE_ITER)

            max_idx = np.argmax(mapped_vals, axis=-1)

            medians[file.split('/')[-1]] = {metric: max_idx.choose(np.rollaxis(np.apply_along_axis(lambda p: p[i], -1, run_splits), -1, 0))
                                            for i, metric in enumerate(self.metrics)}

            if self.settings:
                medians[file.split('/')[-1]]['setting'] = max_idx.choose(
                    np.rollaxis(settings, -1, 0))

        return medians


class ResultsInterpreter:
    """Interprets results saved by Experiment."""

    def __init__(self, files):
        """
        Initializes the interpreter.

        :param files: List of files
        """
        self.learners = []
        self.metrics = []
        self.result = {}
        self.dodge_results = {}
        self.merge_method = None
        self.files = files

    def _read_file(self, filename) -> dict:
        """
        Populates list of learners and metrics, returning the dict in the file read.

        :param filename: File to read
        :return: dict read
        """
        with open(filename, "r") as f:
            line = f.readline()
        line = ast.literal_eval(line)
        self.learners = list(line.keys())
        self.metrics = list(line[self.learners[0]].keys())
        return line

    def get_medians(self):
        """Prints the median results of each treatment"""
        for file in self.files:
            print(file.split('/')[-1].split('.')[0])
            print('=' * len(file.split('/')[-1].split('.')[0]))
            line = self._read_file(file)
            for key in line.keys():
                print(' ', key)
                print(' =' * len(key))

                for metric in self.metrics:
                    print('   ', metric, '\b:', round(
                        median(line[key][metric]), 2))

    def with_dodge(self, i: DODGEInterpreter, merge_method=None):
        """
        Adds DODGE to the result list

        :param i: A DODGEInterpreter object
        :param merge_method: A function describing which files' results will be merged. It is
            passed two arguments: the ResultsInterpreter filename and the
            DODGEInterpreter filename.
        """
        dodge_results = i.interpret()
        self.dodge_results = dodge_results
        self.merge_method = merge_method

        if self.merge_method is None:
            self.merge_method = lambda r, d: r.split(
                "/")[-1] == d.split("/")[-1]
        return self

    def interpret(self):
        """Compares results for each metric using a Scott-Knott test"""
        lines = {}

        # Combine results
        for file in self.files:
            line = self._read_file(file)
            for key in line.keys():
                lines[key + "-" + file.split("/")[-1]] = line[key].copy()

        if len(self.dodge_results.keys()) == 0:
            merge = False
        else:
            merge = True

        result = {}

        for metric in self.metrics:
            print(metric)
            print("=" * len(metric))
            for learner in self.learners:
                for file in self.files:
                    # Add to result
                    key = learner + "-" + file.split("/")[-1]
                    result[key] = lines[key][metric]

            if merge:
                for r_file in self.files:
                    for d_file in self.dodge_results.keys():
                        if self.merge_method(r_file, d_file) and metric in self.dodge_results[d_file]:
                            key = '_dodge-' + r_file.split("/")[-1]
                            result[key] = self.dodge_results[d_file][metric]

            Rx.show(Rx.sk(Rx.data(**result)))
            print()
        self.result = result
