# PREDICTING THE RISE OF SEA LEVEL BY 2050 USING AI
### DESCRIPTION
This project leverages advanced machine learning techniques to predict global sea level anomalies up to the year 2050. The approach combines historical satellite-derived sea level data with synthetic scenario-based features to produce multiple predictive outcomes under different environmental scenarios. By incorporating a Long Short-Term Memory (LSTM) neural network, this project showcases a robust, nonlinear time-series modeling approach to climate forecasting. 

### Objectives 
1. Develop a pipeline to process raw sea level anomaly data from NetCDF files. 
2. Train a Long Short-Term Memory (LSTM) neural network for prediction. 
3. Demonstrate the impact of different environmental scenarios on predictions. 
4. Provide an extensible framework for incorporating real-world climate drivers.

### Dataset 
Source: Satellite-derived NetCDF files containing monthly sea level anomaly (SLA) data. 

Variables Used: SLA (averaged globally) and a synthetic feature simulating climate forcing. 

### PROGRAM
````
pip install xarray netCDF4 pandas matplotlib

import os #to interact with the file system
import xarray as xr #xarray -to handle netCDF files
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting

# Define the folder path containing .nc files
folder_path = r"C:\Users\ritik\Downloads\SEA"
all_data = []

# Loop through all .nc files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".nc"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_path}")

        # Open the NetCDF file
        data = xr.open_dataset(file_path)

        # Extract sea level data and average over latitude and longitude
        if 'sla' in data.variables:  # Ensure 'sla' (sea level anomaly) exists
            sea_level = data['sla'].mean(dim=['latitude', 'longitude'])
            all_data.append(sea_level)
        else:
            print(f"Variable 'sla' not found in file: {file_name}")

# Combine all data into a single time series
if all_data:  # Ensure there's data to concatenate
    combined_sea_level = xr.concat(all_data, dim='time')

    # Plot the combined data
    plt.figure(figsize=(10, 6))
    combined_sea_level.plot()
    plt.title("Combined Global Sea Level (1993-Present)")
    plt.xlabel("Time")
    plt.ylabel("Sea Level Anomaly (m)")
    plt.grid()
    plt.show()
else:
    print("No valid data found in the provided .nc files.")

import pandas as pd #pandas for data handling
from sklearn.preprocessing import MinMaxScaler #normalize data for use in ML models
import numpy as np #numerical operations

# Convert combined sea level data to a pandas DataFrame
df = combined_sea_level.to_dataframe().reset_index()
df = df[['time', 'sla']]  # Keep only time and sea level anomaly
df.rename(columns={'time': 'Date', 'sla': 'Sea_Level_Anomaly'}, inplace=True)

# Set Date as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Resample to yearly averages
df = df.resample('YE').mean()

# Normalize the data for LSTM
scaler = MinMaxScaler()
df['Sea_Level_Anomaly'] = scaler.fit_transform(df[['Sea_Level_Anomaly']])

print(df.head())

# Prepare sequences for LSTM
sequence_length = 5
X, y = [], []

for i in range(len(df) - sequence_length):
    X.append(df['Sea_Level_Anomaly'].iloc[i:i+sequence_length].values)
    y.append(df['Sea_Level_Anomaly'].iloc[i+sequence_length])

X = np.array(X)
y = np.array(y)

# Split into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape for LSTM (add a third dimension for features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
pip install tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Predict future sea levels
future_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=28, freq='Y')
X_future = X_test[-1].reshape(1, sequence_length, 1)  # Start with the last known sequence
future_predictions = []

for _ in future_years:
    prediction = model.predict(X_future)[0, 0]
    future_predictions.append(prediction)
    # Update the input sequence for the next prediction
    X_future = np.roll(X_future, -1, axis=1)
    X_future[0, -1, 0] = prediction

# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

import matplotlib.pyplot as plt

# Plot historical, test, and future predictions
plt.figure(figsize=(12, 6))

# Historical data
plt.plot(df.index, scaler.inverse_transform(df[['Sea_Level_Anomaly']]), label="Historical Data", color="blue")

# Testing data predictions
test_years = df.index[-len(y_test):]
plt.plot(test_years, scaler.inverse_transform(y_test.reshape(-1, 1)), label="Testing Predictions", color="green")

# Future predictions
plt.plot(future_years, future_predictions, label="Future Predictions (2023-2050)", color="red", linestyle='--')

plt.title("Sea Level Predictions (1993-2050)")
plt.xlabel("Year")
plt.ylabel("Sea Level Anomaly (m)")
plt.legend()
plt.grid()
plt.show()

for year, level in zip(future_years, future_predictions):
    print(f"Year {year.year}: Predicted Sea Level = {level:.2f} m")

import pandas as pd
import numpy as np

# Define future years
future_years = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), periods=28, freq='Y')

# Start with the last known sequence
X_future = X_test[-1].reshape(1, sequence_length, 1)
future_predictions = []

# Generate predictions for each future year
for _ in range(len(future_years)):
    prediction = model.predict(X_future)[0, 0]
    future_predictions.append(prediction)
    # Update the input sequence with the new prediction
    X_future = np.roll(X_future, -1, axis=1)
    X_future[0, -1, 0] = prediction

# Inverse transform the predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

# Create a DataFrame for future predictions
future_df = pd.DataFrame({'Year': future_years.year, 'Predicted_Sea_Level_Anomaly': future_predictions})
print(future_df)


````
### OUTPUT
<img width="353" alt="S5" src="https://github.com/user-attachments/assets/70c7bf0f-2a96-4859-ac67-ffb0af12ac98" />
<img width="139" alt="S4" src="https://github.com/user-attachments/assets/c9cfa43e-a135-4296-9094-69642d1c416c" />
<img width="311" alt="S6" src="https://github.com/user-attachments/assets/321bd780-102a-4ee7-af42-747b78021a8e" />
<img width="394" alt="S3" src="https://github.com/user-attachments/assets/0ab8d6c4-aab8-4195-88c0-eb5849870517" />
<img width="196" alt="S2" src="https://github.com/user-attachments/assets/cc62580a-1c88-4fa5-906a-78216d980077" />
<img width="122" alt="S1" src="https://github.com/user-attachments/assets/96bfed3b-0338-4e66-92f3-ec18a2400f2c" />



### RESULT
This project demonstrates the application of machine learning to sea level prediction. By incorporating scenario-based forecasting, it provides a flexible and extensible framework for analyzing the potential impact of climate change on sea level rise. 

