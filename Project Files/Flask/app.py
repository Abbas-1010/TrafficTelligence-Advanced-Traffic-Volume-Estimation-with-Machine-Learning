from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the encoder and model
encoder = joblib.load('encoder.pkl')
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = request.form['weather']
        holiday = request.form['holiday']
        date_str = request.form['date']
        time_str = request.form['time']

        # Convert date and time to datetime features
        date = datetime.strptime(date_str, '%Y-%m-%d')
        hour = datetime.strptime(time_str, '%H:%M').hour
        day_of_week = date.weekday()
        month = date.month
        is_weekend = 1 if day_of_week in [5, 6] else 0

        # Prepare input DataFrame
        row = pd.DataFrame([{
            'temp': temp,
            'rain': rain,
            'snow': snow,
            'weather': weather,
            'holiday': holiday,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend
        }])

        # Transform and predict
        X_encoded = encoder.transform(row)
        prediction = model.predict(X_encoded)[0]

        return render_template('index.html', prediction_text=f'Predicted Traffic Volume: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
