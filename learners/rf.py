from sklearn.ensemble import RandomForestClassifier
import random
from learners.learner import Learner


class RandomForest(Learner):
    """Random forest classifier"""
    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(RandomForest, self).__init__(*args, **kwargs)

        self.learner = RandomForestClassifier()
        if isinstance(self.random, bool) and self.random:
            self.criterion = random.choice(['gini', 'entropy'])
            self.estimators = random.randint(10, 100)
            self.learner = RandomForestClassifier(criterion=self.criterion, n_estimators=self.estimators)
        elif isinstance(self.random, dict):
            for key in self.random.keys():
                setattr(self, key, self.random[key])
