from sklearn.tree import DecisionTreeClassifier
from raise_utils.learners.learner import Learner


class DecisionTree(Learner):
    """Decision tree learner"""
    def __init__(self, weighted=False, *args, **kwargs):
        """Initializes the classifier."""
        super(DecisionTree, self).__init__(*args, **kwargs)

        if weighted:
            self.learner = DecisionTreeClassifier(class_weight="balanced")
        else:
            self.learner = DecisionTreeClassifier()
        self.random_map = {
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"]
        }
        self._instantiate_random_vals()
