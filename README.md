# Machine Learning Project: Naive Bayes Classifier

## Table of Contents

* [Project Description](#project-description)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Data Description](#data-description)
* [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
* [Modeling](#modeling)
* [Evaluation](#evaluation)
* [Summary](#summary)
* [Authors](#authors)
* [References](#references)

---

## Project Description

The goal of this project is to classify two datasets using the **Naive Bayes Classifier** algorithm:

* **Iris Dataset:** Classifying iris flower species.
* **Mushroom Dataset:** Classifying mushrooms as edible or poisonous.

The project includes exploratory data analysis (EDA), model building, and performance evaluation.

The algorithm was implemented in Python using Jupyter Notebook, along with libraries such as scikit-learn, pandas, matplotlib, seaborn, and numpy.

---

## Project Structure

The project includes six main files:

* **`eda_iris.ipynb`** – Contains initial exploratory data analysis for the Iris dataset, including visualizations, interpretations, and conclusions.
* **`evaluation_iris.ipynb`** – Contains model evaluation for the Iris dataset, using the classifier and analyzing its performance.
* **`GaussianNaiveBayesClassifier`** – Implementation of the Gaussian Naive Bayes classifier for the Iris dataset.
* **`eda_mushrooms.ipynb`** – Initial exploratory analysis of the Mushroom dataset and preparation of the data for modeling.
* **`evaluation_mushrooms.ipynb`** – Evaluation of the model on the Mushroom dataset, including usage of the classifier and statistical analysis of the results.
* **`MultinomialNaiveBayesClassifier`** – Implementation of the Multinomial Naive Bayes classifier for the Mushroom dataset.

---

## Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/igorpie1705/NKB.git
   cd NKB
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To use the project, run the following commands:

1. Launch and explore the `EDA` notebooks:

   ```bash
   jupyter notebook eda_iris.ipynb
   jupyter notebook eda_mushrooms.ipynb
   ```
2. Launch and explore the `evaluation` notebooks:

   ```bash
   jupyter notebook evaluation_iris.ipynb
   jupyter notebook evaluation_mushrooms.ipynb
   ```

---

## Data Description

The project uses two datasets:

**Iris Dataset:**

* Features: sepal length & width, petal length & width.
* Target: classify iris species (Setosa, Versicolor, Virginica).

**Mushroom Dataset:**

* Features: various characteristics of cap, stem, odor, etc.
* Target: classify mushrooms as edible or poisonous.

---

## Exploratory Data Analysis (EDA)

During EDA, the following steps were performed:

* Visualizations using histograms, box plots, and pair plots to understand feature distributions and relationships.
* Checking for missing values and their potential impact on modeling.
* Analysis of feature distributions and identification of potential outliers.

This analysis provided better insight into the data structure and its suitability for classification.

---

## Modeling

The Naive Bayes Classifier was used for classification. It is a probabilistic algorithm that applies Bayes' Theorem under the assumption of feature independence. More details can be found in the [References](#references) section.

Modeling included:

* Data preparation, such as normalization of numerical features and encoding of categorical variables (for the Mushroom dataset).
* Splitting the data into training and test sets in a 70:30 ratio to ensure reliable model evaluation.
* Training and testing of models on independent data subsets.

These steps provided a solid foundation for evaluating the algorithm’s performance.

---

## Evaluation

The following metrics were used to evaluate the models:

* **Accuracy:** The ratio of correctly classified samples to the total number of samples in the test set.
* **Confusion Matrix:** Detailed classification results, including false positives and false negatives.
* **Probability Matrix:** Calculation of the probability of assignment to each category, presented as a histogram.

The evaluation results showed that the Naive Bayes Classifier is particularly effective for classification tasks on well-prepared datasets.

---

## Summary

This project demonstrates how effectively a Naive Bayes Classifier can be applied to classify diverse datasets. The analysis and evaluation results confirm that even simple algorithms with strong mathematical foundations can be effective and have broad practical applications, especially in classification tasks.

---

## Authors

This project was developed by:

* **Igor Piesik**
* **Maciej Sitny**

---

## References

1. scikit-learn documentation: [naive-bayes](https://scikit-learn.org/1.5/modules/naive_bayes.html)
2. Kaggle: [naive-bayes-classifier-in-python](https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python)
3. Mushroom dataset: [mushroom-classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
4. StatQuest channel: [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
