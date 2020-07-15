import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout, GlobalMaxPooling1D, Embedding

from sklearn.model_selection import train_test_split

import ml_utility

# Reading raw IMDB review rather than tokenized words from keras.datasets
# Two columns [review, sentiment]
reviews = pd.read_csv("IMDBDataset.csv", delimiter=",")


# Preprocess reviews by separating sentiment (value we're trying to predict)
# from the review itself
# Tokenize words using keras' tokenizer and pad sequences to MAXLEN
# Tokenizer will convert words into an integer value so our embedding
# layer in the model will convert that into a learned word vector
def preprocess(reviews, max_words, maxlen):
    sentiment = (reviews["sentiment"] == "positive").astype("int64")

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(reviews.review)

    sequences = tokenizer.texts_to_sequences(reviews.review)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)

    return padded_sequences, sentiment.to_numpy()


# Create 1D convolutional model
# Can experiement this with different RNN layers such as LSTM and GRU
# Probably won't be as good since sentiment analysis focuses on key
# words within the review rather than the whole review
# It'll also take much longer time to train with LSTM and GRU
def create_conv1d_model(max_words, maxlen):
    model = Sequential()

    model.add(Embedding(max_words, 64, input_length=maxlen))
    model.add(Conv1D(32, 3, activation="relu"))
    model.add(Conv1D(64, 5, activation="relu"))
    # model.add(GlobalMaxPooling1D())
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["binary_accuracy"])

    model.summary()
    return model


# Tune-able hyper parameters
# MAX_WORDS is the top MAX_WORDS number of vocabulary to extract
# MAXLEN is the length of the sequence, how many words per sequence
MAX_WORDS = 10000
MAXLEN = 100
BATCH_SIZE = 32
EPOCHS = 10

# Split data into training and testing set
# Validation set will be generated by model.fit
X, Y = preprocess(reviews, MAX_WORDS, MAXLEN)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = create_conv1d_model(MAX_WORDS, MAXLEN)

history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_split=.2)

ml_utility.plot_history(history, EPOCHS, ["binary_accuracy", "loss"])