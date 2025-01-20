import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("./mushrooms.csv")

class MultinomialNaiveBayesClassifier:
    def __init__(self, X_train,X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def fit(self):
        #na początku muszę obliczyć prawdopodobieństwo apriori dla klas jadalny i trujący
        self.classes = np.unique(self.y_train)
        self.priors = {}
        for cls in self.classes:
            self.priors[cls] = np.sum(self.y_train == cls) / len(self.y_train)

        #teraz będę zliczał wystąpienia każdej cechy w każdej klasie
        self.feature_counts = {}
        for cls in self.classes:
            self.feature_counts[cls] = {}
            for feature in self.X_train.columns:
                self.feature_counts[cls][feature] = {}
                for value in np.unique(self.X_train[feature]):
                    self.feature_counts[cls][feature][value] = (self.X_train[feature][self.y_train == cls] == value).sum()

        #mamy zliczone wystąpienia tych cech, więc teraz obliczam prawdopodobieństwo warunkowe
        self.feature_probs = {}
        for cls in self.classes:
            self.feature_probs[cls] = {}
            for feature in self.X_train.columns:
                self.feature_probs[cls][feature] = {}
                for value in np.unique(self.X_train[feature]):
                    self.feature_probs[cls][feature][value] = (self.feature_counts[cls][feature][value]+1) / (np.sum(self.y_train == cls) + len(np.unique(self.X_train[feature])))

    def predict(self,X):
        probs=self.predict_proba(X)
        return [max(prob, key=prob.get) for prob in probs]

    def predict_proba(self, X):
        result = []
        for line in X.values:
            probs = {}
            for cls in self.classes:
                probs[cls] = self.priors[cls]
                for feature, value in zip(X.columns, line):
                    probs[cls] *= self.feature_probs[cls][feature].get(value, 1e-10)
            total = sum(probs.values())
            for cls in self.classes:
                probs[cls] /= total
            result.append(probs)
        return result

    def score(self,X,y):
        predictions = self.predict(X)
        correct = sum(pred == true for pred, true in zip(predictions, y))
        return correct / len(y)


