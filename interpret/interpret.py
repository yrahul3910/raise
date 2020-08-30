from interpret.sk import Rx
from statistics import median
from typing import Union, Callable
import numpy as np


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
        self.files = files

    def _read_file(self, filename) -> dict:
        """
        Populates list of learners and metrics, returning the dict in the file read.

        :param filename: File to read
        :return: dict read
        """
        with open(filename, "r") as f:
            line = f.readline()
        line = eval(line)
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

    def compare(self):
        """Compares results for each metric using a Scott-Knott test"""
        lines = {}

        # Combine results
        for file in self.files:
            line = self._read_file(file)
            for key in line.keys():
                lines[key + "-" + file.split("/")[-1]] = line[key].copy()

        result = {}

        for metric in self.metrics:
            print(metric)
            print("=" * len(metric))
            for learner in self.learners:
                for file in self.files:
                    # Add to result
                    key = learner + "-" + file.split("/")[-1]
                    result[key] = lines[key][metric]

            Rx.show(Rx.sk(Rx.data(**result)))
            print()
        self.result = result


class DODGEInterpreter:
    """Interprets the results of DODGE-generated files"""

    def __init__(self, files: list = [], max_by: Union[None, int, Callable[..., str]] = None, exclude_cols: list = []) -> None:
        """
        Initializes the interpreter.

        Arguments:
        ==========
        :param files - A list of files to be interpreted.
        :param max_by - Either a None, int, or Callable. If None, defaults to
                        maximizing the first entry, the metric maximized by DODGE.
                        If int, maximizes by the index specified.
                        If callable, maximizes by the function passed.
        :param exclude_cols - List of column indices to exclude

        Returns:
        ========
        :return DODGEInterpreter object
        """
        self.files = files
        if max_by is None:
            self.max_by = 0
        else:
            self.max_by = max_by
        self.exclude_cols = exclude_cols

    def interpret(self) -> np.ndarray:
        DODGE_ITER = 30
        for file in self.files:
            print(file)
            print('=' * len(file))

            with open(file, 'r') as f:
                lines = f.readlines()

            lines = [eval(line.split(':')[1])
                     for line in lines if line.startswith('iter')]
            n_runs = int(len(lines) // DODGE_ITER)
            n_metrics = len(lines[0]) - len(self.exclude_cols)

            lines = np.array(lines)
            lines = np.delete(lines, self.exclude_cols, -1)

            assert lines.shape == (n_runs * DODGE_ITER, n_metrics)

            run_splits = lines.reshape(
                (n_runs, DODGE_ITER, n_metrics))

            if isinstance(self.max_by, int):
                mapped_vals = np.apply_along_axis(
                    lambda x: x[self.max_by], axis=1, arr=run_splits)
            elif callable(self.max_by):
                mapped_vals = np.apply_along_axis(
                    lambda x: self.max_by(x), axis=1, arr=run_splits)

            assert mapped_vals.shape == (n_runs, n_metrics)

            max_idx = np.argmax(mapped_vals, axis=-2)
            return np.median(max_idx.choose(np.rollaxis(run_splits, -2, 0)), axis=0)
