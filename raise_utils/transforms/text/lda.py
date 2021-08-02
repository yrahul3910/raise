import random

from sklearn.decomposition import LatentDirichletAllocation

from raise_utils.data import Data


class LDA:
    def __init__(self, random=True):
        self.random = random
        self.transformer = None

    def fit_transform(self, data: Data):
        a, b, c = random.randint(10, 50), random.random(), random.random()
        d, e, f = random.uniform(0.51, 1.0), random.uniform(1, 50), random.choice([150, 180, 210, 250, 300])
        lda = LatentDirichletAllocation(n_components=a, doc_topic_prior=b, topic_word_prior=c,
                                        learning_decay=d, learning_offset=e, batch_size=f,
                                        max_iter=100, learning_method='online')
        lda.fit_transform(data.x_train)
        self.transformer = lda

    def transform(self, x_test):
        if self.transformer is None:
            raise AssertionError("fit_transform must be called before transform.")

        return self.transformer.transform(x_test)
