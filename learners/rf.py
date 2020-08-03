from sklearn.ensemble import RandomForestClassifier
import random
from learners.learner import Learner


class RandomForest(Learner):
    """
    Random forest classifier
    """
    def __init__(self, *args, **kwargs):
        super(RandomForest, self).__init__(*args, **kwargs)
        if self.random:
            self.criterion = random.choice(['gini', 'entropy'])
            self.estimators = random.randint(10, 100)
            self.learner = RandomForestClassifier(criterion=self.criterion, n_estimators=self.estimators)
        else:
            self.learner = RandomForestClassifier()
