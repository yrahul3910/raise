from sklearn.naive_bayes import GaussianNB
from raise_utils.learners.learner import Learner


class NaiveBayes(Learner):
    """The Naive Bayes classifier."""
    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(NaiveBayes, self).__init__(*args, **kwargs)
        self.learner = GaussianNB()
