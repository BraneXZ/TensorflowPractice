# Please clone/download the repo here
# https://github.com/jbrownlee/Datasets
# The dataset I will be using for this is the daily-min-temperatures.csv

import pandas as pd
import numpy as np

from keras.layers import Conv1D, LSTM, GRU, Dense, Flatten
from keras.models import Sequential

from python_files import ml_utility

# Read dataset and assign the Date column as our index column
datasets_path = "../datasets/daily-min-temperatures.csv"
temperature_dataset = pd.read_csv(datasets_path, index_col="Date")

# Feel free to uncomment these and try it out in a jupyter notebook
# to get a feel of the data
# temperature_dataset.head()
# temperature_dataset.plot()


# Splits the time series into a set window size
# Shuffle the resulting data to reduce sequence bias
def windowed_series(data, window_size, shuffle=True):
    X = []
    Y = []

    data = data.to_numpy().flatten()
    for window_index in range(len(data) - window_size - 1):
        x = []
        for i in range(window_index, window_index + window_size):
            x.append([data[i]])
        # Prediction would be the next value after the window
        y = data[window_index+window_size]

        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    if shuffle:
        shuffle_index = np.arange(len(X))
        np.random.shuffle(shuffle_index)
        X = X[shuffle_index]
        Y = Y[shuffle_index]

    return X, Y


# Tuneable hyper parameters
WINDOW_SIZE = 120
EPOCHS = 200
BATCH_SIZE = 16
LOSS = "huber_loss"

train_x, train_y = windowed_series(temperature_dataset, WINDOW_SIZE)

# Simple dense model to set baseline
simple_model = Sequential()
simple_model.add(Flatten(input_shape=[WINDOW_SIZE, 1]))
simple_model.add(Dense(10, activation="relu"))
simple_model.add(Dense(1))  # 1 output unit with no activation because
                            # this is a regression task

simple_model.compile(optimizer="rmsprop",
                     loss=LOSS,
                     metrics=["mae"])

# history= simple_model.fit(train_x, train_y,
#                           epochs=EPOCHS,
#                           batch_size=BATCH_SIZE,
#                           validation_split=.2,
#                           verbose=2)

# ml_utility.plot_history(history, EPOCHS, ["loss", "mae"])


# RNN model using LSTM/GRU
model = Sequential()
model.add(Conv1D(64, WINDOW_SIZE, input_shape=[None, 1], activation="relu")) # Conv1D start makes this much faster
model.add(LSTM(64, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="rmsprop",
              loss=LOSS,
              metrics=["mae"])

history = model.fit(train_x, train_y,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=.2,
                    verbose=2)


ml_utility.plot_history(history, EPOCHS, ["loss", "mae"])