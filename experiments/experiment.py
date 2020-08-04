import time

from metrics.metrics import ClassificationMetrics
from transform.transform import Transform
import numpy as np
import random
import string


class Experiment:
    """
    Base class for experiments
    """
    def __init__(self, json: dict):
        """"
        Initializes the experiment.
        :param json: A dict containing the parameters. See docs for details.
        """
        self.n_runs: int = json.get('runs', 20)
        self.transforms: list = json.get('transforms', [])  # list of str
        self.metrics: list = json.get('metrics', ['accuracy'])  # list of str
        self.random: bool = json.get('random', False)
        self.learners: list = json['learners']
        self.log: str = json.get('log_path', './log/')
        self.data: list = json['data']  # list
        self.name: str = json.get('name', ''.join(random.choices(string.ascii_letters, k=10)))

    def run(self):
        """
        Runs the experiment
        :return:
        """
        print("Running experiment", self.name)
        print('=' * len("Running experiment" + str(self.name)))

        start_time = time.time()

        # Accumulate the data
        if len(self.data) > 1:
            data = self.data[0]
            for i in range(1, len(self.data)):
                data += self.data[i]
            self.data = data

        # Apply transforms
        for t in self.transforms:
            transform = Transform(t, random=self.random)
            transform.apply(self.data)

        results = {l.__name__: {m: [] for m in self.metrics} for l in self.learners}

        for i in range(self.n_runs):
            print(" Run #", str(i) + ':', flush=True)
            print('=' * len("Run #" + str(i) + ':'))

            # Initialize the learners
            for learner in self.learners:
                learner.set_data(self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test)
                learner.fit()

            # Make predictions
            predictions = []
            for learner in self.learners:
                predictions.append(learner.predict(self.data.x_test))

                # Evaluate
                metric = ClassificationMetrics(self.data.y_test, predictions)
                if "popt20" in self.metrics:
                    metric.add_data(np.concatenate((self.data.x_train, self.data.y_train), axis=1))
                values = metric.get_metrics()
                for j, m in enumerate(self.metrics):
                    results[learner.__name__][m].append(values[j])

        end_time = time.time()
        print("Experiment completed in", str(end_time - start_time), "seconds. Writing results to file.")

        with open(self.name, 'w') as f:
            f.write(str(results))

        print("Results written.\nDone.")
