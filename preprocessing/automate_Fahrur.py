import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse
import os

class DataPreprocessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def process(self):
            print(f"Memuat data dari {self.input_path}...")
            df = pd.read_csv(self.input_path)

            print("Memulai Preprocessing...")
            # 1. Handling Missing Values (Spasi di TotalCharges)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
            df.dropna(inplace=True)

            # 2. Menghapus Data Duplikat (Tambahan)
            df.drop_duplicates(inplace=True)

            # 3. Drop kolom tidak penting
            if 'customerID' in df.columns:
                df.drop('customerID', axis=1, inplace=True)

            # 4. Encoding Kategorikal
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

            # 5. Scaling
            scaler = StandardScaler()
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            df[num_cols] = scaler.fit_transform(df[num_cols])

            # 6. Menyimpan data
            print(f"Menyimpan data bersih ke {self.output_path}...")
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)
            print("Preprocessing Selesai!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automasi Preprocessing Telco Churn")
    parser.add_argument('--input', type=str, required=True, help="Path ke dataset raw")
    parser.add_argument('--output', type=str, required=True, help="Path ke dataset bersih ...")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(args.input, args.output)
    preprocessor.process()