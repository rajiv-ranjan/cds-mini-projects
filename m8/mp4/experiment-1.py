import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Step 1: Load and Prepare the Data ---
print("Step 1: Loading and preparing data...")
# Load the dataset
df = pd.read_csv("PRSA_Data_Dingling_20130301-20170228.csv")

# Create a datetime column and set it as the index
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
df.set_index('datetime', inplace=True)

# Resample to daily frequency and forward-fill missing values
daily_data = df['PM2.5'].resample('D').mean().fillna(method='ffill')
print("Data loaded and prepared successfully.")
print(daily_data.head())
print("-" * 30)


# --- Step 2: Visualize the Time Series Data ---
print("Step 2: Visualizing the time series data...")
plt.figure(figsize=(12, 6))
plt.plot(daily_data)
plt.title('Daily PM2.5 Levels in Beijing (2013-2017)')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration (ug/m^3)')
plt.grid(True)
plt.show()
print("-" * 30)


# --- Step 3: Check for Stationarity (Dickey-Fuller Test) ---
print("Step 3: Checking for stationarity...")
# Perform the Augmented Dickey-Fuller test
result = adfuller(daily_data)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Interpret the results
if result[1] > 0.05:
    print("\nSeries is not stationary, differencing is required.")
    # Apply differencing
    daily_data_diff = daily_data.diff().dropna()
    # Re-run the test on differenced data
    result_diff = adfuller(daily_data_diff)
    print('\nAfter differencing:')
    print('ADF Statistic: %f' % result_diff[0])
    print('p-value: %f' % result_diff[1])
else:
    print("\nSeries is stationary.")
    daily_data_diff = daily_data

# Plot the differenced data
plt.figure(figsize=(12, 6))
plt.plot(daily_data_diff)
plt.title('Differenced Daily PM2.5 Levels')
plt.xlabel('Date')
plt.ylabel('Differenced PM2.5 Concentration')
plt.grid(True)
plt.show()
print("-" * 30)


# --- Step 4: ACF and PACF Plots ---
print("Step 4: Plotting ACF and PACF...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(daily_data_diff, ax=ax1, lags=40)
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(daily_data_diff, ax=ax2, lags=40)
ax2.set_title('Partial Autocorrelation Function (PACF)')
plt.show()
print("-" * 30)


# --- Step 5: Split Data, Build & Fit ARIMA Model ---
print("Step 5: Building and fitting the ARIMA model...")
# Split data into training (80%) and testing (20%) sets
train_size = int(len(daily_data_diff) * 0.8)
train, test = daily_data_diff[0:train_size], daily_data_diff[train_size:]

# Define and fit the ARIMA model (p=5, d=1, q=1)
model = ARIMA(train, order=(5, 1, 1))
model_fit = model.fit()

# Print the model summary
print(model_fit.summary())
print("-" * 30)


# --- Step 6: Make Predictions and Evaluate ---
print("Step 6: Making predictions and evaluating the model...")
# Make predictions on the test set
predictions = model_fit.forecast(steps=len(test))
predictions_series = pd.Series(predictions, index=test.index)

# Calculate Mean Squared Error
mse = mean_squared_error(test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Values', color='blue')
plt.plot(predictions_series.index, predictions_series, label='Predicted Values', color='red', linestyle='--')
plt.title('ARIMA Model Forecast vs Actual Data')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration')
plt.legend()
plt.grid(True)
plt.show()
print("\nScript finished.")