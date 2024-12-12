import pandas as pd
from scipy import stats

def load_data(file_path):
    print(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    print(f"Data loaded. Shape: {data.shape}")
    return data

def check_missing_values(data):
    missing_values = data.isnull().sum()
    print("Missing values in the dataset:")
    print(missing_values)
    return missing_values

def detect_outliers(data):
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    print(f"Numerical columns for outlier detection: {numerical_cols}")
    
    z_scores = stats.zscore(data[numerical_cols])
    print("Z-scores calculated for numerical columns.")
    
    outliers = data[(z_scores > 3).any(axis=1) | (z_scores < -3).any(axis=1)]
    print(f"Outliers detected: {outliers.shape[0]} rows")
    return outliers

def clean_data(data, outliers):
    print(f"Cleaning data by removing {outliers.shape[0]} outlier rows.")
    data_clean = data[~data.index.isin(outliers.index)]
    print(f"Cleaned data shape: {data_clean.shape}")
    return data_clean

def remove_duplicates(data):
    print(f"Removing duplicates. Original data shape: {data.shape}")
    data_cleaned = data.drop_duplicates()
    print(f"Data after removing duplicates: {data_cleaned.shape}")
    return data_cleaned

def consistency_in_dates_sentiment(data):
    print("Converting 'Dates' column to datetime format...")
    data['Dates'] = pd.to_datetime(data['Dates'], format='%d-%m-%Y', errors='coerce')
    print(f"Converted 'Dates' column. Any NaT values due to conversion? {data['Dates'].isna().sum()}")

    future_dates = data[data['Dates'] > pd.Timestamp.now()]
    print(f"Found {future_dates.shape[0]} future dates in the dataset.")
    return future_dates

def consistency_in_dates_price(data):
    print("Converting 'Dates' column to datetime format...")
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
    print(f"Converted 'Dates' column. Any NaT values due to conversion? {data['Date'].isna().sum()}")

    future_dates = data[data['Date'] > pd.Timestamp.now()]
    print(f"Found {future_dates.shape[0]} future dates in the dataset.")
    return future_dates

def check_data_types_price(data):
    print("Checking data types for consistency...")
    expected_types = {
        'Date': 'datetime64[ns]',
        'Open': 'float64',
        'High': 'float64',
        'Low': 'float64',
        'Close': 'float64',
        'Adjusted Close': 'float64',
    }
    
    inconsistencies = check_inconsistencies(data, expected_types)
    print(f"Data type inconsistencies found: {len(inconsistencies)}")
    for inconsistency in inconsistencies:
        print(inconsistency)
    return inconsistencies

def check_data_types_sentiment(data):
    print("Checking data types for consistency...")
    
    expected_types = {
        'Dates': 'datetime64[ns]',
        'Price Direction Up': 'int64',
        'Price Direction Constant': 'int64',
        'Price Direction Down': 'int64',
        'Asset Comparision': 'int64',
        'Past Information': 'int64',
        'Future Information': 'int64',
        'Price Sentiment': 'object',
    }
    
    inconsistencies = check_inconsistencies(data, expected_types)
    print(f"Data type inconsistencies found: {len(inconsistencies)}")
    
    for inconsistency in inconsistencies:
        print(inconsistency)
    
    return inconsistencies

def check_inconsistencies(data, expected_types):
    inconsistencies = []
    for column, expected_type in expected_types.items():
        if column in data.columns:
            actual_type = str(data[column].dtype)
            if actual_type != expected_type:
                inconsistencies.append(f"Data type inconsistency for column '{column}': Expected {expected_type}, Found {actual_type}")
    return inconsistencies
