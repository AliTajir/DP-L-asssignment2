# %%
pip install pandas numpy scikit-learn tensorflow yfinance


# %%
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load data from the CSV file
data = pd.read_csv('AAPL.csv')

# Ensure the date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort data by date
data = data.sort_values('Date')

# Handle missing values (if any)
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

# Define the look-back window for time series data
look_back = 30  # You can adjust this as needed

# Create sequences of data for training
X, y = [], []
for i in range(len(data) - look_back):
    X.append(data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i:i+look_back].values)
    y.append(data['Close'].iloc[i+look_back])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build an LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 5)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# %%
import matplotlib.pyplot as plt

# Train the model and capture training history
history = model.fit(X_train, y_train, epochs=10, batch_size=64)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the actual vs. predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Price', color='blue')
plt.plot(y_pred, label='Predicted Price', color='red')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plot the training loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Training Loss Over Time')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



