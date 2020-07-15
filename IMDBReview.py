from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Dropout, Embedding, Dense, Flatten, SimpleRNN, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


NUM_WORDS = 10000
MAXLEN = 100
EMBEDDING_LENGTH = 64
EPOCHS = 20
CALLBACKS = [
    EarlyStopping(monitor="val_binary_accuracy",
                  patience=1),
    ModelCheckpoint("IMDBModel.h5")
]

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

x_train = pad_sequences(x_train, maxlen=MAXLEN)
x_test = pad_sequences(x_test, maxlen=MAXLEN)

print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

def conv1D_model():
    model = Sequential()
    model.add(Embedding(NUM_WORDS, EMBEDDING_LENGTH, input_length=MAXLEN))
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
    model.add(Embedding(NUM_WORDS, EMBEDDING_LENGTH, input_length=MAXLEN))
    model.add(SimpleRNN(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    model.summary()
    return model


def lstm_model():
    model = Sequential()
    model.add(Embedding(NUM_WORDS, EMBEDDING_LENGTH, input_length=MAXLEN))
    model.add(LSTM(64, activation="relu", return_sequences=True, recurrent_dropout=.2, dropout=.2))
    model.add(LSTM(64, activation="relu", recurrent_dropout=.2, dropout=.2))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    model.summary()
    return model


model = lstm_model()

history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    validation_split=.2,
                    batch_size=32)


def plot_history(history, train_metric, val_metric):
    training = history.history[train_metric]
    validation = history.history[val_metric]

    plt.plot(range(1, EPOCHS+1), training, label=f"Training {train_metric}")
    plt.plot(range(1, EPOCHS + 1), validation, label=f"Validation {val_metric}")
    plt.legend()
    plt.title(f"{train_metric} vs {val_metric}")


print(f"Test accuracy: {model.evaluate(x_test, y_test)}")
plt.figure(1)
plot_history(history, "loss", "val_loss")
plt.figure(2)
plot_history(history, "binary_accuracy", "val_binary_accuracy")
plt.show()