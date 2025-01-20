# Projekt Machine Learning: Naiwny Klasyfikatore Bayesowy

## Spis treści

- [Projekt Machine Learning: Naiwny Klasyfikatore Bayesowy](#projekt-machine-learning-naiwy-klasyfikator-bayesowy)
  - [Spis treści](#spis-treści)
  - [Opis projektu](#opis-projektu)
  - [Struktura projektu](#struktura-projektu)
  - [Instalacja](#instalacja)
  - [Sposób użycia](#sposób-użycia)
  - [Opis danych](#opis-danych)
  - [Eksploracyjna analiza danych (EDA)](#eksploracyjna-analiza-danych-eda)
  - [Modelowanie](#modelowanie)
  - [Ewaluacja](#ewaluacja)
  - [Podsumowanie](#podsumowanie)
  - [Autorzy](#autorzy)
  - [Referencje](#referencje)

---

## Opis projektu

Celem projektu jest klasyfikacja dwóch zbiorów danych przy użyciu algorytmu **Naiwnego Klasyfikatora Bayesowego**:

- **Zbiór danych Irysów:** Klasyfikacja gatunków kwiatów irysa.
- **Zbiór danych Grzybów:** Klasyfikacja grzybów jako jadalne lub trujące.

Projekt obejmuje eksploracyjną analizę danych (EDA), budowę modelu oraz jego ocenę.

Do implementacji algorytmu wykorzystano język Python w środowisku Jupyter Notebook, wraz z bibliotekami takimi jak scikit-learn, pandas, matplotlib, seaborn oraz numpy.

---

## Struktura projektu

W projekcie znajduje się 6 głównych plików:

- **`eda_iris.ipynb`** – przechowuje wstępną eksploracyjną analizę danych (EDA) dla zbioru danych Iris. Zawiera wizualizacje oraz interpretacje i wnioski.
- **`evaluation_iris.ipynb`** – zawiera ewaluację modelu dla zbioru Iris. Wykorzystuje algorytm klasyfikacji oraz ocenia jego wydajność.
- **`GaussianNaiveBayesClassifier`** – Implementacja naiwnego klasyfikatora bayesowego (Gaussian Naive Bayes) dla zbioru danych Iris.
- **`eda_mushroom.ipynb`** –
- **`evaluation_mushroom.ipynb`** –
- **`MultinomialNaiveBayesClassifier`** –

---

## Instalacja

Aby skonfigurować projekt, wykonaj następujące kroki:

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/igorpie1705/NKB.git
   cd NKB
   ```
2. Zainstaluj wymagane zależności:

   ```bash
   pip install -r requirements.txt
   ```

---

## Sposób użycia

Aby korzystać z projektu uruchom następujące komendy:

1. Uruchom i zapoznaj się z notatnikami `EDA` (Exploratory Data Analysis):
   ```bash
   jupyter notebook eda_iris.ipynb
   jupyter notebook eda_mushroom.ipynb
   ```
2. Uruchom i zapoznaj się z notatnikami `evaluation`
   ```bash
   jupyter notebook evaluation_iris.ipynb
   jupyter notebook evaluation_mushroom.ipynb
   ```

---

## Opis danych

Projekt wykorzystuje dwa zbiory danych:

**Zbiór Irysów**:

- Cechy: długość i szerokość działki kielicha oraz płatka.

- Cel: klasyfikacja gatunku irysa (Setosa, Versicolor, Virginica).

**Zbiór grzybów:**

- Cechy: różne cechy kapelusza, trzonu, zapachu itp.

- Cel: klasyfikacja grzybów jako jadalne lub trujące.

---

## Eksploracyjna analiza danych (EDA)

Podczas eksploracyjnej analizy danych wykonano:

- Wizualizację danych za pomocą histogramów, wykresów pudełkowych i par (pair plots), aby zrozumieć rozkład cech oraz relacje między nimi.
- Sprawdzenie brakujących wartości w zbiorach danych i ich potencjalnego wpływu na modelowanie.
- Analizę rozkładu cech oraz identyfikację potencjalnych wartości odstających.

Dzięki tej analizie lepiej zrozumiano strukturę danych oraz ich przydatność do klasyfikacji.

---

## Modelowanie

Do klasyfikacji danych wykorzystano Naiwny Klasyfikator Bayesowski. Jest to algorytm probabilistyczny, który stosuje Twierdzenie Bayesa z założeniem, że cechy są niezależne. Szczegóły działania można zobaczyć wchodząc w linki w zakładce [Referencje](#Referencje).

Modele zostały stworzone uwzględniając:

- Przygotowanie danych, w tym normalizację cech liczbowych oraz kodowanie zmiennych kategorycznych w przypadku zbioru danych grzybów.
- Podział danych na zbiory treningowe i testowe w stosunku 70:30, aby zapewnić wiarygodną ocenę modelu.
- Proces trenowania modeli oraz ich testowania na niezależnym zbiorze danych.

Dzięki temu uzyskano wystarczające podstawy do przeprowadzenia ewaluacji wydajności algorytmu.

---

## Ewaluacja

Do oceny modeli wykorzystano następujące metryki:

- Dokładność (Accuracy): Mierzona jako stosunek poprawnie sklasyfikowanych przykładów do wszystkich przykładów w zbiorze testowym.
- Macierz pomyłek: Analiza szczegółowych wyników klasyfikacji, w tym liczby fałszywych pozytywów i fałszywych negatywów.
- Macierz prawdopodobieństw: Obliczenie prawdopodobieństwa przypisania do danej kategorii i przedstawienie wyników na histogramie.

Wyniki ewaluacji pokazały, że Naiwny Klasyfikator Bayesowski jest szczególnie skuteczny w zadaniach klasyfikacyjnych na dobrze przygotowanych danych.

---

## Podsumowanie

Projekt pokazuje, jak skutecznie można zastosować Naiwny Klasyfikator Bayesowski do klasyfikacji różnorodnych zbiorów danych. Wyniki analizy i ewaluacji potwierdzają, że nawet proste algorytmy na bazie matematyki mogą być skuteczne i mieć szerokie zastosowanie w praktyce, zwłaszcza w zadaniach klasyfikacyjnych

---

## Autorzy

Projekt został opracowany przez:

- **Igor Piesik**
- **Maciej Sitny**

---

## Referencje

1. Dokumentacja scikit-learn: [naive-bayes](https://scikit-learn.org/1.5/modules/naive_bayes.html)
2. Kaggle: [naive-bayes-classifier-in-python](https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python)
3. Baza danych grzybów: [mushroom-classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
4. Kanał StatQuest: [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA&ab_channel=StatQuestwithJoshStarmer)
