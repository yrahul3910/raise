from sklearn.ensemble import RandomForestClassifier
from learners.learner import Learner


class RandomForest(Learner):
    """Random forest classifier"""
    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(RandomForest, self).__init__(*args, **kwargs)

        self.learner = RandomForestClassifier()
        self.random_map = {
            "criterion": ["gini", "entropy"],
            "n_estimators": (10, 100)
        }
        self._instantiate_random_vals()
