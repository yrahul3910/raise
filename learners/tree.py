from sklearn.tree import DecisionTreeClassifier
from learners.learner import Learner
import random


class DecisionTree(Learner):
    """Decision tree learner"""
    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(DecisionTree, self).__init__(*args, **kwargs)

        self.learner = DecisionTreeClassifier()
        if isinstance(self.random, bool) and self.random:
            self.criterion = random.choice(['gini', 'entropy'])
            self.splitter = random.choice(['best', 'random'])
            self.learner = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter)
        elif isinstance(self.random, dict):
            for key in self.random.keys():
                setattr(self, key, self.random[key])
