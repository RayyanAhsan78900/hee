import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# API and base URL for fetching data
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data for a given timeframe
def fetch_data(start_date, end_date):
    params = {
        "access_key": api_key,
        "base": "USD",
        "symbols": "TIN",
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(f"{base_url}/timeseries", params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('success', False):
            return data.get("rates", {})
        else:
            print("API request failed:", data.get("error", {}).get("info"))
            return None
    else:
        print("Error fetching data:", response.status_code)
        return None

# Fetch data in smaller timeframes and combine results
start_dates = ["2024-07-01", "2024-07-16"]
end_dates = ["2024-07-15", "2024-07-31"]
all_data = {}
for start_date, end_date in zip(start_dates, end_dates):
    data = fetch_data(start_date, end_date)
    if data:
        all_data.update(data)

if all_data:
    df = pd.DataFrame.from_dict(all_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.reset_index().rename(columns={"index": "ds", "TIN": "y"})
    df = df[["ds", "y"]]
    print(df.head(30))  # Show more rows to verify data integrity
else:
    print("No data fetched.")

# Check for anomalies and outliers
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], marker='o', linestyle='-')
plt.title('Price of Tin Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Check for missing values and fill them if any
print("Missing values before filling:\n", df.isnull().sum())
df.fillna(method='ffill', inplace=True)
print("Missing values after filling:\n", df.isnull().sum())

# Decompose the time series if there is enough data
if len(df) >= 60:  # Requires at least 2 cycles for period=30
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(df['y'], model='additive', period=30)
    result.plot()
    plt.show()
else:
    print("Insufficient data for seasonal decomposition. Only", len(df), "observations available.")

# Initialize the Prophet model with adjusted parameters
model = Prophet(
    changepoint_prior_scale=0.1,  # Adjust based on performance
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Fit the model
model.fit(df)

# Create a dataframe for future dates
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)

# Visualize the forecast
fig = model.plot(forecast)
plt.title('Prophet Forecast with Improved Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Evaluate the model using cross-validation
df_cv = cross_validation(model, initial='14 days', period='7 days', horizon='7 days')
df_performance = performance_metrics(df_cv)
print(df_performance)

# Train-test split for evaluation
split_index = int(0.8 * len(df))
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Create a new Prophet model for the training set
train_model = Prophet(
    changepoint_prior_scale=0.1,
    yearly_seasonality=True,
    weekly_seasonality=True
)

# Fit the new model on the training set
train_model.fit(train_df)

# Make predictions on the test set
future_test = train_model.make_future_dataframe(periods=len(test_df), include_history=False)
forecast_test = train_model.predict(future_test)

# Evaluate the model
y_test = test_df['y'].values
y_pred = forecast_test['yhat'].values

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Calculate prediction accuracy as a percentage
accuracy = 100 - (mae / y_test.mean() * 100)
print(f"Prediction Accuracy: {accuracy:.2f}%")

# Visualize the forecast vs actual values
plt.figure(figsize=(10, 6))
plt.plot(test_df['ds'], y_test, label='Actual')
plt.plot(test_df['ds'], y_pred, label='Predicted')
plt.legend()
plt.title('Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Consider alternative models: ARIMA
# For ARIMA, ensure the series is stationary
# Check for stationarity
result = adfuller(df['y'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Fit ARIMA model (adjust p, d, q as necessary)
arima_model = ARIMA(df['y'], order=(5,1,0))  # Example order
arima_result = arima_model.fit()

# Forecast using ARIMA
arima_forecast = arima_result.get_forecast(steps=15)
arima_conf_int = arima_forecast.conf_int()
arima_pred = arima_forecast.predicted_mean

# Plot ARIMA forecast
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Historical')
plt.plot(pd.date_range(start=df['ds'].iloc[-1], periods=16, freq='D')[1:], arima_pred, label='ARIMA Forecast')
plt.fill_between(pd.date_range(start=df['ds'].iloc[-1], periods=16, freq='D')[1:],
                 arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title('ARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Function to predict the price for a specific future date using Prophet
def predict_price_for_date(date_str):
    future_date = pd.to_datetime(date_str)
    future = pd.DataFrame({'ds': [future_date]})
    forecast = model.predict(future)
    return forecast['yhat'].values[0]

# Get user input for the prediction date
user_input = input("Enter the date for which you want to predict the price (YYYY-MM-DD): ")
predicted_price = predict_price_for_date(user_input)
print(f"The predicted price of tin on {user_input} is: {predicted_price}")

