import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

import ml_utility

# Read data in as a pandas dataframe and set date time as the index
# This dataset contains 14 features and we're trying to predict temperature in Celsius
dataset = pd.read_csv("jena_climate_2009_2016.csv")
dataset.set_index("Date Time", inplace=True)
print(dataset.columns)

# Run this if you'd like to see the temperature data in the dataset
# plt.figure(0)
# plt.plot(range(0, len(dataset)), dataset["T (degC)"].values)
# plt.title("Temperature")
# plt.show()
# plt.clf()


# Tune-able hypter parameters
# HISTORY_SIZE dictates the size of the time series to look at for prediction
# TARGET_SIZE determines the number of predictions to make
# FEATURE_COLUMNS can be every feature or just a selected few for training
# LABEL_COLUMN feature we're trying to predict in the future
HISTORY_SIZE = 144
TARGET_SIZE = 5
# FEATURE_COLUMNS = ["T (degC)", "Tpot (K)"]
FEATURE_COLUMNS = dataset.columns
LABEL_COLUMN = "T (degC)"
LABEL_COLUMN_INDEX = dataset.columns.get_loc(LABEL_COLUMN)
BATCH_SIZE = 32
EPOCHS = 10
CALLBACKS = [
    ModelCheckpoint("lstm_temperature_forecasting_model.h5")
]

# Splits dataset into train, test split with 80/10/10 split as default
def prepare_train_validation_test_split(data, train_size=0.8, val_size=0.1, test_size=0.1):
    train_index = int(len(data) * train_size)
    val_index = train_index + int(len(data) * val_size)

    # Note that we do not shuffle our dataset because this is a time series where
    # the order of our data matters
    train = data[:train_index]
    val = data[train_index:val_index]
    test = data[val_index:]

    # Normalize data by subtracting mean and std from our training dataset
    train_mean = train.mean()
    train_std = train.std()

    # Note that we're normalizing both validation and testing dataset with the
    # mean and std from the training set rather than using their own mean and std
    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std

    return train, val, test


# Create a generator that will have HISTORY_SIZE number of input data to our model
# and TARGET_SIZE number of output future prediction
# Parameters here are described above on their specific usage
def multivariate_data_generator(data,
                                history_size,
                                target_size,
                                feature_columns,
                                label_column,
                                batch_size):
    # Infinite loop here because tensorflow expects generator to be a infinite loop
    # so it can train on multiple epochs
    while True:
        X = []
        Y = []

        for index in range(len(data) - history_size - target_size):
            # Input only contain features listed in FEATURE_COLUMNS with size HISTORY_SIZE
            # Output only contain LABEL_COLUMN with size TARGET_SIZE
            x = data.iloc[index:index + history_size][feature_columns].to_numpy(dtype="float64")
            y = data.iloc[index + history_size:index + history_size + target_size][label_column]

            X.append(x)
            Y.append(y)

            # Yield when size reaches BATCH_SIZE
            if len(Y) == batch_size:
                yield np.array(X), np.array(Y)
                X = []
                Y = []


train_data, val_data, test_data = prepare_train_validation_test_split(dataset)

train_generator = multivariate_data_generator(train_data,
                                              HISTORY_SIZE,
                                              TARGET_SIZE,
                                              FEATURE_COLUMNS,
                                              LABEL_COLUMN,
                                              BATCH_SIZE)
val_generator = multivariate_data_generator(val_data,
                                            HISTORY_SIZE,
                                            TARGET_SIZE,
                                            FEATURE_COLUMNS,
                                            LABEL_COLUMN,
                                            BATCH_SIZE)
test_generator = multivariate_data_generator(test_data,
                                             HISTORY_SIZE,
                                             TARGET_SIZE,
                                             FEATURE_COLUMNS,
                                             LABEL_COLUMN,
                                             BATCH_SIZE)


# Build a LSTM model
def create_LSTM_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(HISTORY_SIZE, len(FEATURE_COLUMNS))))
    model.add(Dense(TARGET_SIZE))
    model.compile(optimizer="rmsprop",
                  loss="mae")

    model.summary()

    return model

# Use this if you want to train using the whole training set (~20min per epoch)
# STEPS_PER_EPOCH = (len(train_data) - HISTORY_SIZE - TARGET_SIZE) // 32
# Or this if you want faster training
STEPS_PER_EPOCH = 100

# Use this if you want to use your trained model for prediction
MODEL_PATH = "lstm_temperature_forecasting_model.h5"
model = load_model(MODEL_PATH)

# model = create_LSTM_model()
# history = model.fit(train_generator,
#                     epochs=EPOCHS,
#                     steps_per_epoch=STEPS_PER_EPOCH,
#                     validation_data=val_generator,
#                     validation_steps= (len(val_data) - HISTORY_SIZE - TARGET_SIZE) // 32,
#                     callbacks=CALLBACKS)

# ml_utility.plot_history(history, EPOCHS, ["loss"])


# Plot the first sample from each batch with their history, future, and prediction
# Can use train/validation/test generator created above
def plot_time_series(generator, num_batch=10):
    for i in range(num_batch):
        x, y = next(generator)
        prediction = model.predict(x)[0]

        plt.figure(i)
        plt.plot(range(-HISTORY_SIZE, 0), x[0, :, LABEL_COLUMN_INDEX], ".-", label="History")
        plt.plot(range(0, TARGET_SIZE), y[0], "rx", label="Future")
        plt.plot(range(0, TARGET_SIZE), prediction, "go", label="Model Prediction")
        plt.legend()
    plt.show()


plot_time_series(train_generator)
