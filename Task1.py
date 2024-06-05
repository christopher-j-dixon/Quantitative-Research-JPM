import os 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md
from statsmodels.tsa.seasonal import seasonal_decompose as sd
from statsmodels.tsa.stattools import adfuller as adf
import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(file_path):
    """
    Load the data from a CSV file and convert the 'Dates' column to datetime format.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        df['Dates'] = pd.to_datetime(df['Dates'])
        return df
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def inspect_data(df):
    """
    Inspect the data for null values, data types, and duplicates, and print summary statistics.

    Args:
        df (pandas.DataFrame): Data to inspect.
    """
    print(df.isnull().sum())
    print(df.dtypes.unique())
    print(df.duplicated().sum())
    print(df.head())
    print(df.tail())
    print(df.describe())

def time_series_plot(dates, prices, label, line_color='blue', line_style='-'):
    """
    Time series plotting function.

    Args:
        dates (pandas.core.series.Series): Dates.
        prices (pandas.core.series.Series): Prices.
        label (str): Label for the plot.
        line_color (str): Line color. Default is 'blue'.
        line_style (str): Line style. Default is '-'.
    """
    sns.set(style='whitegrid')
    plt.plot(dates, prices, label=label, color=line_color, linestyle=line_style)
    plt.xlabel("Date (YYYY-MM)")
    plt.ylabel("Price ($)")
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(md.MonthLocator(interval=3)) # interval=3 to showcase seasonality
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.legend()

def plot_original_data(df):
    """
    Plot the original data.

    Args:
        df (pandas.DataFrame): Data to plot.
    """
    plt.figure(figsize=(12,6))
    plt.title('Natural Gas Prices (2020-10-31) - (2024-09-30)')
    time_series_plot(df['Dates'], df['Prices'], 'Natural gas prices')

def decompose_seasonal_data(df):
    """
    Perform seasonal decomposition on the data and plot the components.

    Args:
        df (pandas.DataFrame): Data to decompose.
    """
    decomposed_df = sd(df['Prices'], period=12)
    decomposed_df.plot()
    plt.suptitle('Seasonal Decomposition of Natural Gas Prices', fontsize=16)

def perform_adf_test(df):
    """
    Perform ADF test for stationarity and print the p-value.

    Args:
        df (pandas.DataFrame): Data to test.
    """
    ADFtest = adf(df['Prices'])
    print(f'ADF Test p-value: {ADFtest[1]}.')

def split_data(df):
    """
    Split the data into training and testing sets.

    Args:
        df (pandas.DataFrame): Data to split.

    Returns:
        pandas.DataFrame, pandas.DataFrame: Training and testing data.
    """
    split = int(0.8 * len(df))
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    return train_df, test_df

def train_sarima_model(train_df):
    """
    Train the SARIMA model on the training data.

    Args:
        train_df (pandas.DataFrame): Training data.

    Returns:
        pm.arima.ARIMA: Trained SARIMA model.
    """
    train_model = pm.auto_arima(train_df['Prices'], seasonal=True, m=12)
    return train_model

def evaluate_model(train_model, test_df):
    """
    Evaluate the trained model on the test data.

    Args:
        train_model (pm.arima.ARIMA): Trained SARIMA model.
        test_df (pandas.DataFrame): Testing data.
    """
    train_pred = train_model.predict(n_periods=len(test_df))
    plt.figure(figsize=(12,6))
    plt.title('Trained vs Actual Prices (2023-12-31) -  (2024-09-30)')
    time_series_plot(test_df['Dates'], test_df['Prices'], 'Actual', line_color='blue')
    time_series_plot(test_df['Dates'], train_pred, 'Predicted', line_color='red')

    print('Model Evaluation on Test Data:')
    print('Model Summary:', train_model.summary())
    print('MAE:', mean_absolute_error(test_df['Prices'], train_pred))
    print('RMSE:', np.sqrt(mean_squared_error(test_df['Prices'], train_pred)))

def train_and_forecast_model(df):
    """
    Train the SARIMA model on the full dataset and generate future predictions.

    Args:
        df (pandas.DataFrame): Full dataset.

    Returns:
        pm.arima.ARIMA, pandas.Series, pandas.DatetimeIndex: Trained model, predictions, and prediction dates.
    """
    model = pm.auto_arima(df['Prices'], seasonal=True, m=12)
    predict = model.predict(n_periods=13)
    predict_dates = pd.date_range(start=df['Dates'].iloc[-1], periods=13, freq='M')

    plt.figure(figsize=(12,6))
    plt.title('Forecasted Prices')
    time_series_plot(df['Dates'], df['Prices'], 'Actual', line_color='blue')
    time_series_plot(predict_dates, predict, 'Forecasted', line_color='green')

    return model, predict, predict_dates

def price_forecast(date, predict, predict_dates):
    """
    Predicts the Gas price for a given date within the forecast range.

    Args:
        date (str): The date in 'YYYY-MM-DD' format.
        predict (pandas.Series): Predicted prices.
        predict_dates (pandas.DatetimeIndex): Prediction dates.

    Returns:
        float or str: Predicted gas price on the given date, or a message if the date is out of range.
    """
    input_date = pd.to_datetime(date)
    prediction_array = predict.values

    if predict_dates.min() <= input_date <= predict_dates.max():
        months_difference = (input_date.year - predict_dates[0].year) * 12 + input_date.month - predict_dates[0].month
        return prediction_array[months_difference]
    elif input_date < predict_dates.min():
        return "Before prediction period. Look to historic price data."
    else:
        return "After prediction period."

# Load and inspect the data
df = load_data('Nat_Gas.csv')
if df is not None:
    inspect_data(df)
    
    # Plot original data
    plot_original_data(df)

    # Perform seasonal decomposition
    decompose_seasonal_data(df)

    # Perform ADF test
    perform_adf_test(df)

    # Split data
    train_df, test_df = split_data(df)

    # Train SARIMA model and evaluate
    train_model = train_sarima_model(train_df)
    evaluate_model(train_model, test_df)

    # Train SARIMA model on full data and forecast
    model, predict, predict_dates = train_and_forecast_model(df)

    # Test price forecast function
    print(price_forecast("2024-09-30", predict, predict_dates))  # Start of the prediction range
    print(price_forecast("2025-05-15", predict, predict_dates))  # Middle of the prediction range
    print(price_forecast("2025-09-30", predict, predict_dates))  # End of the prediction range
    print(price_forecast("2024-09-15", predict, predict_dates))  # A date before the prediction range
    print(price_forecast("2025-10-01", predict, predict_dates))  # A date after the prediction range
    
    # Show plots
    plt.show()
