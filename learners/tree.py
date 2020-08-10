from sklearn.tree import DecisionTreeClassifier
from learners.learner import Learner


class DecisionTree(Learner):
    """Decision tree learner"""
    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(DecisionTree, self).__init__(*args, **kwargs)

        self.learner = DecisionTreeClassifier()
        self.random_map = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"]
        }
        self._instantiate_random_vals()
