from math import sqrt, pi, exp, log
from collections import defaultdict
import numpy as np

# Rozkład normalny
def nd(x, sd, mean):
    root = (1 / (sqrt(2 * pi) * sd))
    exponent = exp(-((x - mean) ** 2) / (2 * sd ** 2))
    return root * exponent


class GaussianNaiveBayesClassifier:
    def fit(self, X, y):
        y = np.array(y)
        self.classes = set(y)
        self.means = defaultdict(list)
        self.sds = defaultdict(list)
        self.priors = {}

        # Policz priors
        total_samples = len(y)
        for cls in self.classes:
            self.priors[cls] = (y == cls).sum() / total_samples
        
        # Oblicz średnią i odchylenie standardowe dla każdej cechy w każdej klasie
        for cls in self.classes:
            cls_data = X[y == cls]
            cls_data_transposed = list(zip(*cls_data))

            for feature in cls_data_transposed:
                mean = sum(feature) / len(feature)
                sd = sqrt(sum((x - mean) ** 2 for x in feature) / len(feature))
                sd = max(sd, 1e-10) # Zabezpieczenie przed zerem (bo nie można użyć logarytmu na 0)
                self.means[cls].append(mean)
                self.sds[cls].append(sd)


    def predict(self, X):
        predictions = []
        # Przejdź po każdej próbce i weź klasę, o najwyższym wyniku
        for sample in X:
            class_probabilities = {}
            for cls in self.classes:
                class_probabilities[cls] = log(self.priors[cls])
                for x, mean, sd in zip(sample, self.means[cls], self.sds[cls]):
                    class_probabilities[cls] += log(nd(x, sd, mean)) # Założenie niezależności cech
            predictions.append(max(class_probabilities, key=class_probabilities.get)) # Reguła decyzyjna
        return predictions


    def predict_proba(self, X):
        result = []
        # Przejdź po każdej próbce i policz prawdopodobieństwo przynależności do kazdej klasy
        for sample in X:
            posteriors = {}
            for cls in self.classes:
                posterior = log(self.priors[cls])
                for x, mean, sd in zip(sample, self.means[cls], self.sds[cls]):
                    posterior[cls] += log(nd(x, sd, mean))
                posteriors[cls] = posterior
                
            total = sum(exp(posteriors[cls]) for cls in posteriors)
            for cls in posteriors:
                posteriors[cls] = exp(posteriors[cls]) / total

            result.append(posteriors)
            
        return result

    def score(self, X, y):
        predictions = self.predict(X)
        correct = 0
        
        for pred, true in zip(predictions, y):
            if pred == true:
                correct += 1

        return correct / len(y)
