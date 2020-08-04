from sklearn.naive_bayes import GaussianNB
from learners.learner import Learner


class NaiveBayes(Learner):
    """The Naive Bayes classifier."""
    def __init__(self, *args, **kwargs):
        """
        Initializes the classifier.
        :param args: Arguments passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(NaiveBayes, self).__init__(*args, **kwargs)
        self.learner = GaussianNB()
