from sklearn.linear_model import LogisticRegression
from learners.learner import Learner


class LogisticRegressionClassifier(Learner):
    """
    The logistic regression classifier
    """
    def __init__(self, *args, **kwargs):
        super(LogisticRegressionClassifier, self).__init__(*args, **kwargs)

        self.learner = LogisticRegression(solver='liblinear')

        self.random_map = {
            "penalty": ['l1', 'l2'],
            "C": [0.1, 1., 10., 100., 1000.]
        }
        self._instantiate_random_vals()
