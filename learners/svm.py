from sklearn.svm import SVC
from learners.learner import Learner
import random


class SVM(Learner):
    """The Support Vector Machine learner"""
    def __init__(self, *args, **kwargs):
        """
        Initializes the classifier.
        :param args: Args passed to Learner
        :param kwargs: Keyword args passed to Learner
        """
        super(SVM, self).__init__(*args, **kwargs)
        print("******\nWARNING: This module is buggy and should NOT be used.\n*****")
        if self.random:
            self.c = random.choice([.01, .1, 1., 10., 100., 1000.])
            self.kernel = random.choice(['poly', 'rbf', 'sigmoid'])
            self.degree = random.randint(2, 4)
            self.gamma = random.random()
            self.learner = SVC(C=self.c, kernel=self.kernel, degree=self.degree, gamma=self.gamma)
        else:
            self.learner = SVC()
