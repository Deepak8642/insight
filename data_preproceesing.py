import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats
def load_data(file_path):
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'windows-1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Loaded data using {encoding} encoding.")
            return df
        except UnicodeDecodeError:
            print(f"Failed to load data using {encoding} encoding. Trying next encoding...")
    raise ValueError("Unable to read the CSV file with available encodings.")

def convert_to_float(df, column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def inspect_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing values in each column:\n", missing_values)
    return missing_values

def preprocess_data(df, drop_threshold=0.1):
    # Identify numeric and non-numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numeric columns: {numeric_columns}")
    print(f"Non-numeric columns: {non_numeric_columns}")

    # Remove non-numeric columns
    df = df[numeric_columns]
    df = df.dropna(axis=1, how='all')
    # Impute missing values for numeric columns
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Z-score outlier detection and removal
    z_scores = np.abs(stats.zscore(df_imputed))
    df_no_outliers = df_imputed[(z_scores < 3).all(axis=1)]

    # Optional: Drop rows with too many missing values (based on threshold)
    if drop_threshold is not None:
        df_no_outliers = df_no_outliers.dropna(thresh=int(drop_threshold * len(df.columns)))

    # Normalize numeric columns
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_no_outliers), columns=df_no_outliers.columns)

    return df_normalized

def main():
    file_path = 'data.csv' #add your file path to preprocess the data 
    df = load_data(file_path)
    
    # Inspect missing values
    inspect_missing_values(df)
    
    df_normalized = preprocess_data(df)

    # Save the preprocessed data
    df_normalized.to_csv('preprocessed_data.csv', index=False)

if __name__ == "__main__":
    main()
