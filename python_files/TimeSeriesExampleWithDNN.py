import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mean_absolute_error

from python_files import ml_utility

# Plot the series using time as x axis and values (series) as y axis
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


# Trend for the time series
def trend(time, slope=0):
    return slope * time


# Random seasonal pattern, can be changed to w/e you want
def seasonal_pattern(season_time):
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))


# Repeats the same pattern at each period
def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern((season_time))


# Generate random noise that'll be added to the series later
def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# Time is 10 years (365 * 10) + 1 day
time = np.arange(10 * 365 + 1, dtype="float32")
# Base line for series value
baseline = 10
# Amplitude for seasonality
amplitude = 40
# Sets the trend of the time series
slope = 0.01
# Determines how noisy the noise is
noise_level = 5

# Creates the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Add noise to the series
series += noise(time, noise_level, seed=51)

# Our total time is 3651, so the first 3000 will be used as training and the rest for validation
# There isn't a test set because we want to evaluate on the validation set and later include
# them for training
# This is because for time series, we'd want to use the most recent data for future prediction
# since they're closely related to the new unseen future data
split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# Set the window size, batch size, and shuffle buffer size
# These can be tuned as hyper parameters to achieve better results/faster performance
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

plot_series(time, series)
plt.show()


# Convert our time series into a numpy array that we can use to pass into our NN model
def windowed_dataset(series, window_size):
    x = []
    y = []
    for i in range(len(series) - window_size):
        x.append(np.array(series[i:i+window_size]))
        y.append(np.array(series[i+window_size]))

    # Shuffle our data to reduce sequence bias
    index = np.arange(len(x))
    np.random.shuffle(index)
    return np.array(x)[index], np.array(y)[index]


# Create a simple NN model with only dense layers
def create_model():
    model = Sequential()
    model.add(Dense(10, input_shape=[window_size], activation="relu"))
    model.add(Dense(20, activation="relu"))
    # Linear activation here since this is a regression task, we're predicting a continous value
    model.add(Dense(1))

    model.compile(optimizer="rmsprop", loss="mse")
    model.summary()

    return model


x_train, y_train = windowed_dataset(x_train, window_size)
model = create_model()

model.fit(x_train, y_train, batch_size=batch_size, epochs=100, verbose=2)

# Forecasting validation set data and compare it with the actual label
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time+window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
plt.show()

print(f"Mean absolute error: {mean_absolute_error(x_valid, results)}")