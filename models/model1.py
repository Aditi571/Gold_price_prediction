import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cred = credentials.Certificate("../firebase.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def fetch_all_data(index_name):
    docs = db.collection(index_name).stream()

    data = [doc.to_dict() for doc in docs]  # Extract data as a list of dictionaries
    df = pd.DataFrame(data)

    logger.info(f"Fetching all records from {index_name}...")
    
    return df

def fetch_and_merge_data(news_index, price_index):
    news_df = fetch_all_data(news_index)
    
    logger.info("News Data (first few records):")
    logger.info(news_df.head())
    
    try:
        logger.info("Converting 'Dates' column to datetime...")
        news_df['Date'] = pd.to_datetime(news_df['Date'], format='%d-%m-%Y', errors='coerce')
        logger.info(f"After conversion, here are the first few rows of dates:")
        logger.info(news_df['Date'].head())
    except Exception as e:
        logger.error(f"Error during date conversion: {e}")
    
    price_df = fetch_all_data(price_index)
    
    logger.info("Price Data (first few records):")
    logger.info(price_df.head())

    try:
        logger.info("Converting 'Date' column to datetime...")
        price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
        logger.info(f"After conversion, here are the first few rows of dates:")
        logger.info(price_df['Date'].head())
    except Exception as e:
        logger.error(f"Error during date conversion: {e}")

    logger.info("Merging news and price data...")
    merged_data = pd.merge(news_df, price_df, left_on='Date', right_on='Date', how='inner')
    
    logger.info(f"Merged Data (first few records):")
    logger.info(merged_data.head())
    
    merged_data = merged_data.sort_values(by='Date')
    print(merged_data.shape)

    return merged_data

def train_random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info(f"Mean Absolute Error: {mae}")
    return mae

def preprocess_data_with_date(merged_df):
    merged_df['Year'] = merged_df['Date'].dt.year
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Day'] = merged_df['Date'].dt.day
    merged_df['DayOfWeek'] = merged_df['Date'].dt.dayofweek
    merged_df['WeekOfYear'] = merged_df['Date'].dt.isocalendar().week

    le = LabelEncoder()
    merged_df['Price Sentiment'] = le.fit_transform(merged_df['Price Sentiment'])

    X = merged_df[['Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
                   'Price Direction Up', 'Price Direction Constant', 
                   'Price Direction Down', 'Asset Comparison', 
                   'Past Information', 'Future Information', 
                   'Price Sentiment']]
    y = merged_df['Adj Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def predict_with_date(model, scaler, date, other_features):
    year = date.year
    month = date.month
    day = date.day
    day_of_week = date.weekday()
    week_of_year = date.isocalendar()[1]

    input_features = [year, month, day, day_of_week, week_of_year] + other_features

    input_scaled = scaler.transform([input_features])

    predicted_price = model.predict(input_scaled)
    return predicted_price[0]

if __name__ == "__main__":
    ES_HOST = "127.0.0.1"
    ES_PORT = 9200
    ES_USER = "elastic"
    ES_PASS = "d3ozO4B7tEJK5Jj4U9*L"
    NEWS_INDEX = "sentiment_data"
    PRICE_INDEX = "price_data"


    merged_data = fetch_and_merge_data(NEWS_INDEX, PRICE_INDEX)

    X_train, X_test, y_train, y_test, scaler = preprocess_data_with_date(merged_data)

    model = train_random_forest_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    input_date = pd.Timestamp('2024-12-01')
    other_features = [1, 0, 0, 1, 0, 1, 0]
    predicted_price = predict_with_date(model, scaler, input_date, other_features)
    logger.info(f"Predicted Adjusted Close for {input_date.date()}: {predicted_price}")

joblib.dump(model, 'random_forest_model.pkl')

joblib.dump(scaler, 'scaler.pkl')

logger.info("Model and scaler saved to disk.")
