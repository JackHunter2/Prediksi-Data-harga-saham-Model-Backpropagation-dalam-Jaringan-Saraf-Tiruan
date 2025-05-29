# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv("TSLA.csv")  # Gantilah dengan dataset saham kamu
df = df[["Open", "High", "Low", "Volume", "Close"]]

# Pisahkan fitur dan target
X = df[["Open", "High", "Low", "Volume"]].values
y = df["Close"].values.reshape(-1, 1)

# Normalisasi
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Bangun model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Callback early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Latih model
model.fit(X_train, y_train, epochs=200, batch_size=16,
          validation_data=(X_test, y_test), callbacks=[early_stopping])

# Simpan model dan scaler
model.save("model_saham.h5")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
