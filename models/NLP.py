from flask import Flask, render_template, request, redirect, url_for
from elasticsearch import Elasticsearch, helpers
import pandas as pd
import joblib


app = Flask(__name__)

es = Elasticsearch(
    hosts=["https://localhost:9200"],
    basic_auth=("elastic", "d3ozO4B7tEJK5Jj4U9*L"),  
    verify_certs=False
)


def insert_price_data(index_name, data):
    document = {
        "Date": data["date"],
        "Adj Close": float(data["adj_close"])
    }

    action = {
        "_op_type": "index", 
        "_index": index_name,  
        "_id": document["Date"],  
        "_source": document  
    }

    helpers.bulk(es, [action])
    print(f"Document with ID {document['Date']} inserted successfully into {index_name}.")

def make_prediction(input_data):
    try:
        model = joblib.load('models/random_forest_model.pkl')
    except Exception as e:
        return f"Model could not be loaded: {e}"

    if model is None:
        return "Model could not be loaded."

    prediction_input = pd.DataFrame([{
        "Date": input_data["Date"],
        "Price Direction Up": input_data["Price Direction Up"],
        "Price Direction Constant": input_data["Price Direction Constant"],
        "Price Direction Down": input_data["Price Direction Down"],
        "Asset Comparision": input_data["Asset Comparision"],
        "Past Information": input_data["Past Information"],
        "Future Information": input_data["Future Information"],
        "Price Sentiment": input_data["Price Sentiment"]
    }])
    
    prediction_input["Date"] = pd.to_datetime(input_data["Date"])

    prediction_input['Year'] = prediction_input['Date'].dt.year
    prediction_input['Month'] = prediction_input['Date'].dt.month
    prediction_input['Day'] = prediction_input['Date'].dt.day
    prediction_input['DayOfWeek'] = prediction_input['Date'].dt.dayofweek
    prediction_input['WeekOfYear'] = prediction_input['Date'].dt.isocalendar().week
    
    prediction_input.drop(columns=["Date"], inplace=True)

    prediction_input['Price Sentiment'] = prediction_input['Price Sentiment'].apply(lambda x: 0 if x == 'negative' else 1)

    prediction_input = prediction_input[[
        'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear',
        'Price Direction Up', 'Price Direction Constant', 
        'Price Direction Down', 'Asset Comparision', 
        'Past Information', 'Future Information', 
        'Price Sentiment'
    ]]

    prediction = model.predict(prediction_input)

    return f"Predicted Price Sentiment: {prediction[0]}"


@app.route('/pricedata', methods=['GET', 'POST'])
def pricedata():
    if request.method == 'POST':
        try:
            date = request.form['date']
            adj_close = float(request.form['adj_close'])
            pd.to_datetime(date, format='%Y-%m-%d', errors='raise')

            price_data = {'date': date, 'adj_close': adj_close}
            
            price_df = pd.DataFrame([price_data])

            insert_price_data('price_data', price_data)
            return redirect(url_for('pricedata'))

        except ValueError as ve:
            return f"Error: Invalid input - {str(ve)}"

    return render_template('pricedata.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        date = request.form['date']
        news = request.form['news']
        price_direction_up = int(request.form['price_direction_up'])
        price_direction_constant = int(request.form['price_direction_constant'])
        price_direction_down = int(request.form['price_direction_down'])
        asset_Comparision = request.form['asset_Comparision']
        past_information = request.form['past_information']
        future_information = request.form['future_information']
        price_sentiment = request.form['price_sentiment']
        
        input_data = {
            "Date": date,
            "News": news,
            "Price Direction Up": price_direction_up,
            "Price Direction Constant": price_direction_constant,
            "Price Direction Down": price_direction_down,
            "Asset Comparision": asset_Comparision,
            "Past Information": past_information,
            "Future Information": future_information,
            "Price Sentiment": price_sentiment
        }

        prediction = make_prediction(input_data)
        print(input_data)
        
        return render_template('prediction_result.html', prediction=prediction)

    return render_template('prediction_form.html')

def insert_sentiment_data(index_name, data):
    fields = [
        "Dates", "News", "Price Direction Up", "Price Direction Constant",
        "Price Direction Down", "Asset Comparision", "Past Information",
        "Future Information", "Price Sentiment"
    ]

    document = dict(zip(fields, data))

    for key in [
        "Price Direction Up", "Price Direction Constant", "Price Direction Down",
        "Asset Comparision", "Past Information", "Future Information"
    ]:
        document[key] = int(document[key])

    action = {
        "_op_type": "index", 
        "_index": index_name,  
        "_id": document["Dates"], 
        "_source": document  
    }

    helpers.bulk(es, [action])

    print(f"Document with ID {document['Dates']} inserted successfully into {index_name}.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        sentiment_data = [
            request.form['dates'],
            request.form['news'],
            request.form['price_direction_up'],
            request.form['price_direction_constant'],
            request.form['price_direction_down'],
            request.form['asset_Comparision'],
            request.form['past_information'],
            request.form['future_information'],
            request.form['price_sentiment']
        ]

        sentiment_df = pd.DataFrame([sentiment_data], columns=[
            "Dates", "News", "Price Direction Up", "Price Direction Constant",
            "Price Direction Down", "Asset Comparision", "Past Information",
            "Future Information", "Price Sentiment"
        ])
        missing_values = check_missing_values(sentiment_df)
        outliers = detect_outliers(sentiment_df)
        clean_data = clean_data(sentiment_df, outliers)
        sentiment_data_cleaned = remove_duplicates(clean_data)

        insert_sentiment_data('sentiment_data', sentiment_data)
        return redirect(url_for('index'))

    return render_template('sentiment.html')


if __name__ == "__main__":
    app.run(debug=True)
