from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Dropout, Embedding, Dense, Flatten, SimpleRNN, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping

import ml_utility

# Tune-able hyper parameters
# MAX_WORDS is the top MAX_WORDS number of vocabulary to extract
# MAXLEN is the length of the sequence, how many words per sequence
# EBEDDING_LENGTH controls the size of the learned word vector
MAX_WORDS = 10000
MAXLEN = 100
EMBEDDING_LENGTH = 64
EPOCHS = 20
BATCH_SIZE = 32
CALLBACKS = [
    EarlyStopping(monitor="val_binary_accuracy",
                  patience=1),
    ModelCheckpoint("IMDBModel.h5")
]

# Load IMDB dataset from keras.dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

# Pad sequences to MAXLEN size
x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

# Print shape here if you'd like to see the shape of our dataset
# print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")


def conv1D_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_LENGTH, input_length=MAXLEN))
    model.add(Conv1D(64, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    model.summary()
    return model


def simple_rnn_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_LENGTH, input_length=MAXLEN))
    model.add(SimpleRNN(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    model.summary()
    return model


def lstm_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, EMBEDDING_LENGTH, input_length=MAXLEN))
    model.add(LSTM(64, activation="relu", return_sequences=True, recurrent_dropout=.2, dropout=.2))
    model.add(LSTM(64, activation="relu", recurrent_dropout=.2, dropout=.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    model.summary()
    return model


# pick one of the models here
# model = simple_rnn_model()
model = conv1D_model()
# model = lstm_model()


history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    validation_split=.2,
                    batch_size=BATCH_SIZE)

ml_utility.plot_history(history, EPOCHS, ["binary_accuracy", "loss"])