from data.data import TextDataLoader
from learners.lstm import TextDeepLearner
from metrics.impl import get_roc_auc

data = TextDataLoader.from_file("pits/pitsA.txt")
learner = TextDeepLearner(epochs=30)
learner.set_data(*data)
learner.fit()

preds = learner.predict(data.x_test)
print(get_roc_auc(data.y_test, preds))
