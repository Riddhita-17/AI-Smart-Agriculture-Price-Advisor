from flask import Flask, render_template, request

import joblib

import numpy as np

app = Flask(__name__)

# LOAD MODEL & ENCODERS

model = joblib.load("price_model.pkl")

le_commodity = joblib.load("le_commodity.pkl")

le_state = joblib.load("le_state.pkl")

le_market = joblib.load("le_market.pkl")

# HOME PAGE

@app.route('/')

def home():

    return render_template("index.html")

# PREDICTION

@app.route('/predict', methods=['POST'])

def predict():

    commodity = request.form['commodity']

    state = request.form['state']

    market = request.form['market']

    year = int(request.form['year'])

    month = int(request.form['month'])

    offered_price = float(request.form['offered_price'])

    # ENCODE INPUTS

    commodity_encoded = le_commodity.transform([commodity])[0]

    state_encoded = le_state.transform([state])[0]

    market_encoded = le_market.transform([market])[0]

    # CREATE FEATURE ARRAY

    features = np.array([[

        commodity_encoded,

        state_encoded,

        market_encoded,

        year,

        month

    ]])

    # PREDICTION

    prediction = model.predict(features)[0]

    prediction = round(prediction, 2)

    # PRICE COMPARISON

    difference = prediction - offered_price

    if difference > 0:

        message = f"⚠ Increase your price by ₹ {round(difference,2)}"

    else:

        message = "✅ Fair Market Price"

    return render_template(

        "index.html",

        prediction=prediction,

        offered_price=offered_price,

        message=message

    )

# RUN APP

if __name__ == '__main__':

    app.run(debug=True)