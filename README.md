COVID-19 Patient Risk Analysis Project
======================================

Opis projektu:
---------------
Ten projekt przedstawia uczenie maszynowe do analizy ryzyka śmierci pacjentów z COVID-19 na podstawie otwartych danych. Pokazuje pełny ML-pipeline: od pobrania danych i analizy wstępnej, przez przetwarzanie danych, po trenowanie modelu i ocenę jego dokładności.

Projekt obejmuje:
- Pobieranie danych z Kaggle (PatientInfo.csv) za pomocą biblioteki kagglehub.
- Czyszczenie i przekształcanie danych (np. '50s' -> 50).
- Wizualizację rozkładu pacjentów i korelacji czynników ryzyka.
- Trenowanie modelu Random Forest i porównanie go z Logistic Regression.
- Wyświetlanie raportów klasyfikacji i dokładności modelu.


Cel projektu:
--------------
- Projekt edukacyjny dotyczący uczenia maszynowego
- Prezentacja pełnego ML-pipeline: od danych do modelu
- Nadaje się do analizy danych medycznych, eksperymentów z modelami i wizualizacji


Struktura projektu:
------------------
```text
Fina-_Pythone/
  main.py                 - Główny skrypt do uruchomienia projektu
  test_main.py            - Testy jednostkowe sprawdzające przetwarzanie danych
  requirements            - Lista zależności Python
  visuals/                - Folder do zapisywania wykresów i wizualizacji
  masz/
    Data.py               - Klasa DataProcessor do pracy z danymi
    modelm.py             - Klasa ModelManager do modeli uczenia maszynowego
```

Instalacja i uruchomienie:
--------------------------
1. Klonowanie repozytorium:
```Bash
    git clone https://github.com/IharSidorkin/Fina-_Pythone.git
   cd Fina-_Pythone
```
3. Instalacja zależności:
   Zaleca się użycie wirtualnego środowiska:
  ```Bash
   python -m venv venv
   source venv/bin/activate   (Linux/macOS)
   venv\Scripts\activate     (Windows)
   pip install -r requirements
```

4. Uruchomienie projektu:
   python main.py

   Projekt wykona następujące kroki:
   1. Pobierze dane z Kaggle (PatientInfo.csv)
   2. Wykona eksploracyjną analizę danych i wizualizację
   3. Przygotuje dane do modelu
   4. Wytrenuje Random Forest i porówna z Logistic Regression
   5. Wyświetli dokładność i raport klasyfikacji

5. Uruchomienie testów:
   python test_main.py
   Sprawdza poprawność działania metody _clean_age() w DataProcessor.

Opis modułów:
-------------
```Bash
DataProcessor (masz/Data.py)
  - load_data()           - pobieranie danych z Kaggle
  - _clean_age()          - czyszczenie kolumny wieku
  - exploratory_analysis()- analiza i wizualizacja danych
  - preprocess_data()     - przygotowanie danych do modeli (kodowanie, podział train/test, skalowanie)
```

ModelManager (masz/modelm.py)
```Bash
  - train_random_forest(X_train, y_train)  - trenowanie Random Forest
  - train_with_optimization(X_train, y_train) - GridSearchCV do wyboru najlepszych parametrów
  - evaluate(X_test, y_test)               - ocena dokładności modelu i raport klasyfikacji
  - compare_models(X_train, y_train, X_test, y_test) - porównanie Random Forest i Logistic Regression
```
Wizualizacje:
--------------
- Wszystkie wykresy są zapisywane w folderze visuals/
- Główny wykres: mapa cieplna korelacji czynników ryzyka wpływających na wynik choroby

Przykład wyjścia:
-----------------
[INFO] Downloading data from Kaggle...
[INFO] Data loaded. Total rows: 5000

Patient distribution by Sex and Outcome:
state    deceased  released
sex
female        200      2300
male          250      2250

[INFO] Preprocessing complete. Training samples: 4000

--- Comparison with Logistic Regression ---
Logistic Regression Accuracy: 0.85
Random Forest Accuracy:       0.91

--- Evaluation Results ---
Accuracy: 0.91
Classification Report:
precision    recall  f1-score   support
...

Wymagania:
-----------
```Bash
- Python 3.8+
- pandas
- matplotlib
- seaborn
- scikit-learn
- kagglehub
- os (moduł wbudowany w Python)
```
