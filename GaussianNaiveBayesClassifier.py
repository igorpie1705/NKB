from math import sqrt, pi, exp, log
from collections import defaultdict

# Rozkład normalny
def nd(x, sd, mean):
    if sd == 0:
        return 1.0 if x == mean else 0.0
    root = (1 / (sqrt(2 * pi) * sd))
    exponent = exp(-((x - mean) ** 2) / (2 * sd ** 2))
    return root * exponent



class GaussianNaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = set(y)
        self.means = defaultdict(list)
        self.sds = defaultdict(list)
        self.priors = {}

        # Policz priors
        for cls in self.classes:
            self.priors[cls] = y.count(cls) / len(y)
            
        # Oblicz średnią i odchylenie standardowe dla każdej cechy w każdej klasie
        for cls in self.classes:
            cls_data = [X[i] for i in range(len(X)) if y[i] == cls]
            cls_data_transposed = list(zip(*cls_data))

            for feature in cls_data_transposed:
                mean = sum(feature) / len(feature)
                sd = sqrt(sum((x - mean) ** 2 for x in feature) / len(feature))
                self.means[cls].append(mean)
                self.sds[cls].append(sd)

    def predict(self, X):
        predictions = []
        for row in X:
            class_probabilities = {}
            for cls in self.classes:
                class_probabilities[cls] = log(self.priors[cls])
                for i in range(len(row)):
                    mean = self.means[cls][i]
                    sd = self.means[cls][i]
                    x = row[i]
                    class_probabilities[cls] += log(nd(x, sd, mean)) # założenie niezależności cech
            predictions.append(max(class_probabilities[cls], key=class_probabilities.get)) # reguła decyzyjna
        return predictions


    def predict_proba(self, X):
        pass


# (method) def fit(
#     X: MatrixLike,
#     y: ArrayLike,
#     sample_weight: ArrayLike | None = None
# ) -> GaussianNB
# Fit Gaussian Naive Bayes according to X, y.

# Parameters
# X : array-like of shape (n_samples, n_features)
#     Training vectors, where n_samples is the number of samples and n_features is the number of features.

# y : array-like of shape (n_samples,)
#     Target values.

# sample_weight : array-like of shape (n_samples,), default=None
#     Weights applied to individual samples (1. for unweighted).

# Returns
# self : object
#     Returns the instance itself.