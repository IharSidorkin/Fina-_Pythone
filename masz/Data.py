import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder  # Заменили на MinMaxScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Class for loading and processing data with POSITIVE values only.
    Using MinMaxScaler and specific encoding to avoid negative correlation.
    """

    def __init__(self):
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        # MinMaxScaler делает все числа от 0 до 1 (никаких минусов!)
        self.scaler = MinMaxScaler()
        self.encoders = {}

    def load_data(self):
        """Step 1: Downloading dynamic data"""
        print("[INFO] Downloading data from Kaggle...")
        path = kagglehub.dataset_download("kimjihoo/coronavirusdataset")
        self.df = pd.read_csv(os.path.join(path, "PatientInfo.csv"))

    def _clean_age(self, dataframe):
        """Cleans '20s' to 20"""
        df_copy = dataframe.copy()
        if 'age' in df_copy.columns:
            df_copy['age'] = df_copy['age'].astype(str).str.replace('s', '', regex=False)
            df_copy['age'] = pd.to_numeric(df_copy['age'], errors='coerce')
        return df_copy

    def exploratory_analysis(self):
        """Step 4: Statistical Analysis (Positive Correlation Focus)"""
        print("\n--- 4. Analysis: Positive Correlation Factors ---")
        temp_df = self._clean_age(self.df).dropna(subset=['age', 'sex', 'state', 'infection_case'])

        corr_df = temp_df.copy()

        # Кодируем пол: Female=0, Male=1
        corr_df['sex_enc'] = LabelEncoder().fit_transform(corr_df['sex'].astype(str))

        # Кодируем источник заражения
        corr_df['source_enc'] = LabelEncoder().fit_transform(corr_df['infection_case'].astype(str))

        # ВАЖНО: Кодируем вручную, чтобы УМЕР = 1 (высокий риск).
        # Теперь рост возраста будет давать рост риска (положительная корреляция)
        corr_df['risk_target'] = corr_df['state'].map({'released': 0, 'deceased': 1})

        plt.figure(figsize=(10, 8))
        # Теперь анализируем связь факторов с РИСКОМ (risk_target)
        relevant_features = ['age', 'sex_enc', 'source_enc', 'risk_target']
        correlation_matrix = corr_df[relevant_features].corr()

        # Рисуем карту в "теплых" тонах. Теперь цифры будут стремиться в плюс!
        sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', fmt=".2f")
        plt.title("Correlation Matrix: Relationship with Mortality Risk")
        plt.show()

    def preprocess_data(self):
        """Step 2, 5, 6: Preprocessing with MinMaxScaler (All Positive)"""
        print("\n[INFO] Preprocessing data into positive range [0, 1]...")

        # Оставляем только завершенные случаи
        self.df = self.df[self.df['state'].isin(['released', 'deceased'])].copy()
        self.df = self._clean_age(self.df)
        self.df['infection_case'] = self.df['infection_case'].fillna('unknown')
        self.df = self.df.dropna(subset=['sex', 'age', 'state'])

        # Векторизация признаков
        le_sex = LabelEncoder()
        self.df['sex'] = le_sex.fit_transform(self.df['sex'].astype(str))

        le_source = LabelEncoder()
        self.df['infection_case'] = le_source.fit_transform(self.df['infection_case'].astype(str))

        # Кодируем цель: 0 - жив, 1 - умер (логика роста риска)
        self.df['target'] = self.df['state'].map({'released': 0, 'deceased': 1})

        X = self.df[['sex', 'age', 'infection_case']]
        y = self.df['target']

        # Разделение данных
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # НОРМАЛИЗАЦИЯ (MinMaxScaler): Теперь X_train будет содержать только числа от 0 до 1.
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print("[INFO] Normalization complete. All values are now positive [0, 1].")