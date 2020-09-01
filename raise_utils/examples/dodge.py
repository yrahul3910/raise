from raise_utils.data import TextDataLoader
from raise_utils.hyperparams import DODGE
from raise_utils.learners import LogisticRegressionClassifier
from raise_utils.learners import NaiveBayes
from raise_utils.learners import RandomForest
from raise_utils.learners import SVM
from raise_utils.learners import DecisionTree


if __name__ == "__main__":
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
