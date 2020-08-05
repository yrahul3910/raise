from data.data import DataLoader
from hyperparams.dodge import DODGE
from learners.logreg import LogisticRegressionClassifier
from learners.nb import NaiveBayes
from learners.rf import RandomForest
from learners.svm import SVM
from learners.tree import DecisionTree


if __name__ == "__main__":
    config = {
        "n_runs": 20,
        "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
        "metrics": ["auc"],
        "random": True,
        "learners": [
                        LogisticRegressionClassifier(random=True, name="lr"),
                        NaiveBayes(random=True, name="nb"),
                        RandomForest(random=True),
                        SVM(random=True),
                        DecisionTree(random=True, name="dt")
                    ] * 50,
        "log_path": "../log",
        "data": [
            DataLoader.from_files("./promise/", ["camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"])
        ],
        "name": "camel-popt20"
    }

    dodge = DODGE(config, verbose=True)
    dodge.optimize()
