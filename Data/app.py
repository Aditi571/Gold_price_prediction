# main.py
import pandas as pd
from dataPreprocessing import load_data, check_missing_values, detect_outliers, clean_data, remove_duplicates, consistency_in_dates_price, consistency_in_dates_sentiment, check_data_types_price,check_data_types_sentiment
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("../firebase.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

file_path = "sentiment.csv"

data = load_data(file_path)

missing_values = check_missing_values(data)

outliers = detect_outliers(data)

data_clean = clean_data(data, outliers)

data_cleaned = remove_duplicates(data_clean)


inconsistencies = check_data_types_sentiment(data_cleaned)


print("Processed Data of sentiment analysis:")
data_cleaned  = data_cleaned.drop(columns=['URL'])
print("Column Names:")
print(data_cleaned.columns.tolist())
print(data_cleaned.head())
sentiment_data=data_cleaned

####################################################################################################################################

file_path = "price.csv"

data = load_data(file_path)
columns_to_keep = ['Date', 'Adj Close']
data = data[columns_to_keep]

missing_values = check_missing_values(data)

outliers = detect_outliers(data)

data_clean = clean_data(data, outliers)

data_cleaned = remove_duplicates(data_clean)

inconsistencies = check_data_types_price(data_cleaned)

print("Processed Data of price analysis:")
print(data_cleaned.head())
price_data=data_cleaned

####################################################################################################################################

# collection_ref_price = db.collection("price_data")
# for index, row in price_data.iterrows():

#     doc_ref = collection_ref_price.document(str(row['Date']))  
#     doc_ref.set({
#         "Date": row['Date'],
#         "Adj Close": row['Adj Close'],
        
#     })

# print("Data upload complete.")
collection_ref_sentiment = db.collection("sentiment_data")
for index, row in sentiment_data.iterrows():

    doc_ref = collection_ref_sentiment.document(str(row['Dates']))  
    doc_ref.set({
        "Date": row['Dates'],  # Mapping Dates to Date field
        "News":row['News'],
        "Price Direction Up": row['Price Direction Up'],  # Mapping Price Direction Up
        "Price Direction Constant": row['Price Direction Constant'],  # Mapping Price Direction Constant
        "Price Direction Down": row['Price Direction Down'],  # Mapping Price Direction Down
        "Asset Comparison": row['Asset Comparision'],  # Mapping Asset Comparison (corrected typo)
        "Past Information": row['Past Information'],  # Mapping Past Information
        "Future Information": row['Future Information'],  # Mapping Future Information
        "Price Sentiment": row['Price Sentiment']  # Mapping Price Sentiment  
    })

print("Data upload complete.")