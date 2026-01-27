import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:
    """
    Requirement #2: Class for loading, analyzing, and processing data.
    All features are scaled to a positive range [0, 1] using MinMaxScaler.
    """

    def __init__(self):
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        # Requirement #5: Normalization (0 to 1 range)
        self.scaler = MinMaxScaler()
        self.encoders = {}

    def load_data(self):
        """Requirement #1: Load data from Kaggle (web source)."""
        print("[INFO] Downloading data from Kaggle...")
        path = kagglehub.dataset_download("kimjihoo/coronavirusdataset")
        filepath = os.path.join(path, "PatientInfo.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File PatientInfo.csv not found in {path}")

        self.df = pd.read_csv(filepath)
        print(f"[INFO] Data loaded. Total rows: {len(self.df)}")

    def _clean_age(self, dataframe):
        """Internal helper to convert text ages ('20s') to numbers (20)."""
        df_copy = dataframe.copy()
        if 'age' in df_copy.columns:
            df_copy['age'] = df_copy['age'].astype(str).str.replace('s', '', regex=False)
            df_copy['age'] = pd.to_numeric(df_copy['age'], errors='coerce')
        return df_copy

    def exploratory_analysis(self):
        """
        Requirement #4: Preliminary Data Analysis (Correlation Matrix).
        Requirement #2: Grouping.
        Saves the correlation heatmap to the 'visuals' folder.
        """
        print("\n--- 4. Statistical Analysis: Clinical Risk Factors ---")

        # Prepare data for analysis (cleaning and dropping missing values)
        temp_df = self._clean_age(self.df).dropna(subset=['age', 'sex', 'state', 'infection_case'])

        # Create directory for plots
        output_dir = "visuals"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Grouping (Requirement #2) - Show distribution by sex and outcome
        print("\nPatient distribution by Sex and Outcome:")
        print(temp_df.groupby(['state', 'sex']).size().unstack())

        # 2. Correlation Matrix Preparation
        corr_df = temp_df.copy()
        le = LabelEncoder()

        # Vectorization for the correlation matrix
        corr_df['source_enc'] = le.fit_transform(corr_df['infection_case'].astype(str))
        corr_df['sex_enc'] = le.fit_transform(corr_df['sex'].astype(str))
        # Positive Logic: 0 - released, 1 - deceased (higher number = higher risk)
        corr_df['risk_target'] = corr_df['state'].map({'released': 0, 'deceased': 1})

        # Plotting Heatmap
        plt.figure(figsize=(10, 8))
        relevant_features = ['age', 'sex_enc', 'source_enc', 'risk_target']
        correlation_matrix = corr_df[relevant_features].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', fmt=".2f")
        plt.title("Correlation Matrix: Factors Influencing Mortality Risk")

        # Save Heatmap to file
        corr_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(corr_path)
        print(f"[INFO] Correlation plot saved to: {corr_path}")

        plt.show()

    def preprocess_data(self):
        """
        Requirements #2, #5, #6: Filtering, Vectorization, Normalization.
        """
        print("\n[INFO] Preprocessing data for machine learning...")

        # Filter only finished cases (Released or Deceased)
        self.df = self.df[self.df['state'].isin(['released', 'deceased'])].copy()
        self.df = self._clean_age(self.df)

        # Handle missing data (Requirement #2: Complementing)
        self.df['infection_case'] = self.df['infection_case'].fillna('unknown')
        self.df = self.df.dropna(subset=['sex', 'age', 'state'])

        # Requirement #6: Vectorization (Encoding categories to numbers)
        for col in ['sex', 'infection_case']:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le

        # Encode target as Risk (0 or 1)
        self.df['target'] = self.df['state'].map({'released': 0, 'deceased': 1})

        # Features (X) and Target (y)
        X = self.df[['sex', 'age', 'infection_case']]
        y = self.df['target']

        # Train/Test Split (80/20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Requirement #5: Normalization (MinMaxScaler)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"[INFO] Preprocessing complete. Training samples: {len(self.X_train)}")
