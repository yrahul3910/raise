from sklearn.linear_model import LogisticRegression
from raise_utils.learners.learner import Learner


class LogisticRegressionClassifier(Learner):
    """
    The logistic regression classifier
    """
    def __init__(self, weighted=False, *args, **kwargs):
        super(LogisticRegressionClassifier, self).__init__(*args, **kwargs)

        if weighted:
            self.learner = LogisticRegression(solver="liblinear", penalty="l1", class_weight="balanced")
        else:
            self.learner = LogisticRegression(solver='liblinear', penalty="l1")

        self.random_map = {
            "penalty": ['l1', 'l2'],
            "C": [0.1, 1., 10., 100., 1000.]
        }
        self._instantiate_random_vals()
