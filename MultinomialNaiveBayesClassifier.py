import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("./mushrooms.csv")

print(df.shape)
print(df.info())

categorical = [var for var in df.columns if df[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

print(df[categorical].isnull().sum())

for var in categorical:
    print(df[var].value_counts())


print(df['stalk-root'].unique())

df['stalk-root'].replace('?', np.nan, inplace=True)

print(df['stalk-root'].unique())

print(df['stalk-root'].value_counts())


X = df.drop(['class'], axis=1)

y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =0)

print(X_train.shape, X_test.shape)


categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']
print(X_train[categorical].isnull().mean())


for df2 in [X_train, X_test]:
    df2['stalk-root'].fillna(X_train['stalk-root'].mode()[0], inplace=True)

print(X_train[categorical].isnull().mean())
print(X_test[categorical].isnull().sum())


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


classifier = MultinomialNaiveBayesClassifier(X_train, X_test, y_train, y_test)
classifier.fit()


classifier.predict_proba(X_test)
# print(classifier.predict(X_test))
print(classifier.score(X_train, y_train))
print(classifier.score(X_test, y_test))

