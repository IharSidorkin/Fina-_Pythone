import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataProcessor:


    def __init__(self):
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.scaler = MinMaxScaler()
        self.encoders = {}

    def load_data(self):
        print("[INFO] Downloading data from Kaggle...")
        path = kagglehub.dataset_download("kimjihoo/coronavirusdataset")
        filepath = os.path.join(path, "PatientInfo.csv")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File PatientInfo.csv not found in {path}")

        self.df = pd.read_csv(filepath)
        print(f"[INFO] Data loaded. Total rows: {len(self.df)}")

    def _clean_age(self, dataframe):
        df_copy = dataframe.copy()
        if 'age' in df_copy.columns:
            df_copy['age'] = df_copy['age'].astype(str).str.replace('s', '', regex=False)
            df_copy['age'] = pd.to_numeric(df_copy['age'], errors='coerce')
        return df_copy

    def exploratory_analysis(self):

        print("\n--- 4. Statistical Analysis: Clinical Risk Factors ---")


        temp_df = self._clean_age(self.df).dropna(subset=['age', 'sex', 'state', 'infection_case'])


        output_dir = "visuals"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        print("\nPatient distribution by Sex and Outcome:")
        print(temp_df.groupby(['state', 'sex']).size().unstack())


        corr_df = temp_df.copy()
        le = LabelEncoder()


        corr_df['source_enc'] = le.fit_transform(corr_df['infection_case'].astype(str))
        corr_df['sex_enc'] = le.fit_transform(corr_df['sex'].astype(str))

        corr_df['risk_target'] = corr_df['state'].map({'released': 0, 'deceased': 1})


        plt.figure(figsize=(10, 8))
        relevant_features = ['age', 'sex_enc', 'source_enc', 'risk_target']
        correlation_matrix = corr_df[relevant_features].corr()

        sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', fmt=".2f")
        plt.title("Correlation Matrix: Factors Influencing Mortality Risk")


        corr_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.savefig(corr_path)
        print(f"[INFO] Correlation plot saved to: {corr_path}")

        plt.show()

    def preprocess_data(self):

        print("\n[INFO] Preprocessing data for machine learning...")


        self.df = self.df[self.df['state'].isin(['released', 'deceased'])].copy()
        self.df = self._clean_age(self.df)


        self.df['infection_case'] = self.df['infection_case'].fillna('unknown')
        self.df = self.df.dropna(subset=['sex', 'age', 'state'])


        for col in ['sex', 'infection_case']:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le


        self.df['target'] = self.df['state'].map({'released': 0, 'deceased': 1})


        X = self.df[['sex', 'age', 'infection_case']]
        y = self.df['target']


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        print(f"[INFO] Preprocessing complete. Training samples: {len(self.X_train)}")
