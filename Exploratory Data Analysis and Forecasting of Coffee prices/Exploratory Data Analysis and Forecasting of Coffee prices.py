# -*- coding: utf-8 -*-
"""

@author: vbalestrieri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Stations, Daily, Point
from datetime import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import yfinance as yf
from numpy.linalg import LinAlgError
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

# Example: Fetching historical prices for coffee futures (symbol may vary)
coffee = yf.download("KC=F", start="2010-01-01", end="2023-12-31")
coffee_prices = coffee['Adj Close']
coffee_volume = coffee['Volume']

def fetch_weather_data(lat, lon, alt=None, start=datetime(2010, 1, 1), end=datetime(2023, 12, 31)):
    # Create a Point for the specified location
    location = Point(lat, lon, alt)
    
    # Initialize Stations object and find nearby stations
    stations = Stations()
    nearby_stations = stations.nearby(lat, lon).fetch()
    
    # Check if there are any stations returned
    if nearby_stations.empty:
        print("No nearby stations found.")
        return pd.DataFrame()

    # Use the first station ID from the nearby stations
    first_station_id = nearby_stations.index[0]
    
    # Fetch daily weather data for the first nearby station
    data = Daily(first_station_id, start, end)
    data = data.fetch()
    
    # Convert index to datetime if necessary
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Resample the data to monthly averages and sums
    weather_monthly = data.resample('D').agg({'tavg': 'mean', 'prcp': 'sum'}).rename(columns={'tavg': 'avg_temp', 'prcp': 'total_precip'})
    
    return weather_monthly

# Define your time period
start = datetime(2010, 1, 1)
end = datetime(2023, 12, 31)

# Fetch data for each region
#for Minas Gerais, Brazil

weather_brazil = fetch_weather_data(lat=-18.10, lon=-44.38)

# For Central Highlands, Vietnam
weather_vietnam = fetch_weather_data(lat=14.67, lon=108.0)

# For Antioquia, Colombia
weather_colombia = fetch_weather_data(lat=6.55, lon=-75.83)

# For Sidamo, Ethiopia
weather_ethiopia = fetch_weather_data(lat=6.0, lon=38.5)

# For Sumatra, Indonesia
weather_indonesia = fetch_weather_data(lat=-0.5897, lon=101.3431)

# For Comayagua, Honduras
weather_honduras = fetch_weather_data(lat=14.4608, lon=-87.6475)

# For Central Region, Uganda
weather_uganda = fetch_weather_data(lat=0.3167, lon=32.5833)

##### Decided to use brazil, colombia and ethipia because of data stability issues

# Combine weather data
combined_weather = weather_brazil.join([weather_vietnam, weather_colombia], how='outer')

weather_brazil = weather_brazil.rename(columns={'avg_temp': 'avg_temp_brazil', 'total_precip': 'total_precip_brazil'})
weather_colombia = weather_colombia.rename(columns={'avg_temp': 'avg_temp_colombia', 'total_precip': 'total_precip_colombia'})
weather_ethiopia = weather_ethiopia.rename(columns={'avg_temp': 'avg_temp_ethiopia', 'total_precip': 'total_precip_ethiopia'})

# Merge with coffee price data
combined_weather = pd.concat([weather_brazil, weather_ethiopia, weather_colombia], axis=1)




# Prepare your dataset (ensure all necessary preprocessing)
if isinstance(coffee_prices, pd.Series):
    coffee_prices = coffee_prices.to_frame('coffee_price')

combined_data = pd.concat([combined_weather, coffee_prices], axis=1)

combined_data['coffee_price'] = combined_data['coffee_price'].fillna(method='ffill')

if isinstance(coffee_volume, pd.Series):
    coffee_volume = coffee_volume.to_frame('coffee_volume')

combined_data = pd.concat([combined_data, coffee_volume], axis=1)

combined_data['coffee_volume'] = combined_data['coffee_volume'].fillna(method='ffill')

combined_data.index = pd.DatetimeIndex(combined_data.index).to_period('D')
model_vars = combined_data.dropna()


def test_stationarity(timeseries, window=12):
    # Convert PeriodIndex to DateTimeIndex for plotting
    if isinstance(timeseries.index, pd.PeriodIndex):
        timeseries_for_plot = timeseries.copy()
        timeseries_for_plot.index = timeseries_for_plot.index.to_timestamp()
    else:
        timeseries_for_plot = timeseries

    # Determine rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries_for_plot, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)

# test the 'Price' series for stationarity
test_stationarity(model_vars['coffee_price'])

# Differencing the series
coffee_price_diff = model_vars['coffee_price'].diff().dropna()

# After differencing, test stationarity again.
test_stationarity(coffee_price_diff)

#transformations
coffee_price_log = np.log(model_vars['coffee_price']).dropna()

# Test stationarity again
test_stationarity(coffee_price_log)

model_vars_diff = model_vars.diff().dropna()



model = VAR(model_vars_diff)

for i in range(1, 11):
    try:
        result = model.fit(i)
        print(f'Lag Order = {i}')
        print('AIC :', result.aic)
        print('BIC :', result.bic)
        print()
    except LinAlgError as e:
        print(f'Failed to fit model with {i} lags: {e}')


#Given the data, lag order 6 appears to be a reasonable choice

lag_order = 6

model_fitted = model.fit(lag_order)
residuals = model_fitted.resid
num_columns = residuals.shape[1]

# Set the number of rows and columns in your subplot grid
nrows = int(np.ceil(np.sqrt(num_columns)))
ncols = int(np.ceil(num_columns / nrows))

# Create a figure and axes with subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 10))

# Flatten axes array if necessary
if num_columns > 1:
    axes = axes.flatten()
else:
    axes = [axes]

# Loop through the columns and create an ACF plot on each subplot
for i, column in enumerate(residuals.columns):
    # Plot the ACF starting from lag 1 (hence lags=range(1, 41))
    plot_acf(residuals[column], lags=range(1, 41), alpha=0.05, ax=axes[i], title=f'Residuals ACF for {column}')
    axes[i].set_title(f'Residuals ACF for {column}')  # Set title to each subplot

# Adjust layout for better fit and display
plt.tight_layout()
plt.show()

maxlag=12
test = 'ssr_chi2test'


def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values are 
    zero, that is, the X does not cause Y can be rejected."""

    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix=grangers_causation_matrix(model_vars_diff, variables = model_vars_diff.columns) 

# Define a custom color palette
custom_cmap = sns.diverging_palette(190, 21, as_cmap=True)

# Plot the heatmap with the custom color palette
sns.heatmap(grangers_causation_matrix, cmap=custom_cmap)
plt.title('Granger Causality')
plt.show()

# Splitting data
train = model_vars_diff.iloc[:-50, :]
test = model_vars_diff.iloc[-50:, :]

# Fitting model on training data
model = VAR(train)
model_fitted = model.fit(lag_order)



# Forecasting
forecast = model_fitted.forecast(train.values[-lag_order:], steps=10)

forecast_df = pd.DataFrame(forecast, index=test.index[:10], columns=test.columns)

train.index = train.index.to_timestamp()
test.index = test.index.to_timestamp()

# Plot the original and forecasted values
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['coffee_price'], label='Original', color='blue')
plt.plot(forecast_df.index, forecast_df['coffee_price'], label='Forecasted', color='red')
plt.xlabel('Date')
plt.ylabel('coffee_price')
plt.title('Forecast vs. Actual')
plt.legend()
last_date = max(train.index)
plt.xlim(last_date - pd.Timedelta(days=60), last_date + pd.Timedelta(days=30))

plt.show()

# Side-by-Side Plot
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(train.index, train['coffee_price'], label='Original', color='blue')
plt.title('Original Coffee Price')
plt.xlabel('Date')
plt.ylabel('Coffee Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(test.index[:10], forecast_df['coffee_price'], label='Forecasted', color='red')
plt.title('Predicted Coffee Price')
plt.xlabel('Date')
plt.ylabel('Coffee Price')
plt.legend()

plt.tight_layout()
plt.show()
