from sklearn.linear_model import LogisticRegression
import random
from learners.learner import Learner


class LogisticRegressionClassifier(Learner):
    """
    The logistic regression classifier
    """
    def __init__(self, *args, **kwargs):
        super(LogisticRegressionClassifier, self).__init__(*args, **kwargs)

        self.learner = LogisticRegression(solver='liblinear')

        if isinstance(self.random, bool) and self.random:
            self.penalty = random.choice(['l1', 'l2'])
            self.c = random.choice([0.1, 1., 10., 100., 1000.])
            self.learner = LogisticRegression(penalty=self.penalty, C=self.c, solver='liblinear')
        elif isinstance(self.random, dict):
            for key in self.random.keys():
                setattr(self, key, self.random[key])
