from sklearn.tree import DecisionTreeClassifier
from learners.learner import Learner
import random


class DecisionTree(Learner):
    """
    Decision tree learner
    """
    def __init__(self, *args, **kwargs):
        super(DecisionTree, self).__init__(*args, **kwargs)
        if self.random:
            self.criterion = random.choice(['gini', 'entropy'])
            self.splitter = random.choice(['best', 'random'])
            self.learner = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter)
        else:
            self.learner = DecisionTreeClassifier()
