"""
An example demonstrating the use of the DODGE hyperparameter optimizer.
In this example, we train on one of the PROMISE repository files, and
test the use of several transforms and learners. DODGE is set to run
for 20 iterations.
"""

from raise_utils.data import TextDataLoader
from raise_utils.hyperparams import DODGE
from raise_utils.learners import LogisticRegressionClassifier
from raise_utils.learners import NaiveBayes
from raise_utils.learners import RandomForest
from raise_utils.learners import SVM
from raise_utils.learners import DecisionTree


if __name__ == "__main__":
    """
    Set up the DODGE configuration object with the desired number of
    iterations, the metrics to print (only the first, i.e., f1 is
    optimized for), and random parameters for the transforms.
    """

    config = {
        "n_runs": 20,
        "transforms": ["tf", "tfidf", "hashing", "lda"] * 30,
        "metrics": ["f1", "pd", "pf", "auc", "prec"],
        "random": True,
        "learners": [
            LogisticRegressionClassifier(random=True, name="lr"),
            NaiveBayes(random=True, name="nb"),
            RandomForest(random=True),
            SVM(random=True),
            DecisionTree(random=True, name="dt")
        ],
        "log_path": "../log",
        "data": [
            TextDataLoader.from_file("./pits/pitsA.txt")
        ],
        "name": "camel-pd-pf"
    }

    """
    This is a hack that adds 50 more copies of the learners. We
    cannot simply multiply the list by 50 since that copies references,
    not the objects.
    """

    for _ in range(50):
        config["learners"].extend([
            LogisticRegressionClassifier(random=True, name="lr"),
            NaiveBayes(random=True, name="nb"),
            RandomForest(random=True),
            SVM(random=True),
            DecisionTree(random=True, name="dt")
        ])

    dodge = DODGE(config)
    dodge.optimize()
