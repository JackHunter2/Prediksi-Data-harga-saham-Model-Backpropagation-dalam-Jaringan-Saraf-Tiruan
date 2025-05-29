# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler
model = load_model("model_saham.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    open_price = 0
    high_price = 0
    low_price = 0

    if request.method == 'POST':
        try:
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            volume = float(request.form['volume'])

            # Normalisasi input
            input_data = np.array([[open_price, high_price, low_price, volume]])
            input_scaled = scaler_X.transform(input_data)

            # Prediksi
            predicted_scaled = model.predict(input_scaled)
            predicted_price = scaler_y.inverse_transform(predicted_scaled)

            prediction = round(predicted_price.flatten()[0], 2)
        except Exception as e:
            print("Error:", e)
            prediction = "Terjadi kesalahan pada input."

    return render_template('index.html',
                           prediction=prediction,
                           open=open_price,
                           high=high_price,
                           low=low_price)

if __name__ == '__main__':
    app.run(debug=True)
