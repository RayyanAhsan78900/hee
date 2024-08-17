from flask import Flask, request, render_template, jsonify
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
import io
import base64

app = Flask(__name__)

# Set up your API and base URL for fetching data
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data for a given timeframe, splitting into chunks if necessary
def fetch_data(start_date, end_date):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    all_data = {}

    while start_date <= end_date:
        current_end_date = min(start_date + timedelta(days=29), end_date)  # 29 days per chunk
        params = {
            "access_key": api_key,
            "base": "USD",
            "symbols": "TIN",
            "start_date": start_date.strftime(date_format),
            "end_date": current_end_date.strftime(date_format)
        }
        response = requests.get(f"{base_url}/timeseries", params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                all_data.update(data.get("rates", {}))
            else:
                return None, f"API request failed: {data.get('error', {}).get('info')}"
        else:
            return None, f"Error fetching data: {response.status_code}"

        start_date = current_end_date + timedelta(days=1)  # Move to the next chunk

    return all_data if all_data else None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_date = request.form['start_date']
        prediction_period = request.form['prediction_period']

        if prediction_period == "6 Months":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=6 * 30)  # Approximate 6 months
        elif prediction_period == "3 Months":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=3 * 30)  # Approximate 3 months
        elif prediction_period == "3 Weeks":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=3)
        elif prediction_period == "1 Week":
            end_date = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(weeks=1)

        end_date_str = end_date.strftime('%Y-%m-%d')
        data, error = fetch_data(start_date, end_date_str)

        if data:
            df = pd.DataFrame.from_dict(data, orient="index")
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "ds", "TIN": "y"})
            df = df[["ds", "y"]]
            df.fillna(method='ffill', inplace=True)

            # Prophet Model
            model = Prophet(
                changepoint_prior_scale=0.1,
                yearly_seasonality=True,
                weekly_seasonality=True
            )
            model.fit(df)

            prediction_days = (end_date - datetime.strptime(start_date, "%Y-%m-%d")).days
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)

            # Plotting Prophet forecast
            fig1 = model.plot(forecast)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Performance metrics
            df_cv = cross_validation(model, initial='14 days', period='7 days', horizon='7 days')
            df_performance = performance_metrics(df_cv)

            # ARIMA Model
            result = adfuller(df['y'])
            is_stationary = result[1] < 0.05
            arima_plot_url = None
            if is_stationary:
                arima_model = ARIMA(df['y'], order=(5, 1, 0))
                arima_result = arima_model.fit()

                arima_forecast = arima_result.get_forecast(steps=prediction_days)
                arima_conf_int = arima_forecast.conf_int()
                arima_pred = arima_forecast.predicted_mean

                plt.figure(figsize=(10, 6))
                plt.plot(df['ds'], df['y'], label='Historical')
                plt.plot(pd.date_range(start=df['ds'].iloc[-1], periods=prediction_days + 1, freq='D')[1:], arima_pred,
                         label='ARIMA Forecast')
                plt.fill_between(pd.date_range(start=df['ds'].iloc[-1], periods=prediction_days + 1, freq='D')[1:],
                                 arima_conf_int.iloc[:, 0], arima_conf_int.iloc[:, 1], color='pink', alpha=0.3)
                plt.legend()
                plt.title('ARIMA Forecast')
                plt.xlabel('Date')
                plt.ylabel('Price')

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                arima_plot_url = base64.b64encode(img.getvalue()).decode()

            return render_template('index.html', plot_url=plot_url, df_performance=df_performance.to_html(),
                                   arima_plot_url=arima_plot_url, is_stationary=is_stationary)

        else:
            return render_template('index.html', error=error)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
