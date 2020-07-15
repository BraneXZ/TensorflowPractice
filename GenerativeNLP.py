import nltk
from nltk.corpus import gutenberg

import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import sys

import ml_utility

# Download gutenberg corpus from nltk
# It'll update if already downloaded
nltk.download("gutenberg")

# Print this ouf to see other corpus in gutenberg and choose w/e you want
# print(gutenberg.fileids())

full_bible_text = gutenberg.raw("bible-kjv.txt").lower()
# Only selecting the first 100,000 characters because this text file
# contains ~4million characters, and it'll take ages to process with LSTM
text = full_bible_text[:100000]

# All present unique characters within this text
# 2 dictionaries to map chars to integer, integer to chars for OHE
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# A generator rather than a list because storing everything in memory might
# blow up your computer
def sequence_generator(MAXLEN, STEPS, BATCH_SIZE):
    # When fitting a generator to models, tensorflow requires the generator
    # to be infinitely iterable
    while True:
        ohe_sequences = []
        ohe_predictions = []
        # Take MAXLEN characters as sequences, step by STEPS so it overlaps
        # a little, but not all. One hot encode sequence by character and
        # one hot encode the next character as the prediction
        # Yields based on batch size
        for i in range(0, len(text) - MAXLEN - STEPS, STEPS):
            ohe_sequence = []
            for j in range(MAXLEN):
                ohe_char = np.zeros(len(chars))
                char = text[i + j]
                ohe_char[char_indices[char]] = 1
                ohe_sequence.append(ohe_char)

            ohe_prediction = np.zeros(len(chars))
            ohe_prediction[char_indices[text[i + MAXLEN]]] = 1

            ohe_sequences.append(ohe_sequence)
            ohe_predictions.append(ohe_prediction)

            if len(ohe_sequences) == BATCH_SIZE:
                yield (np.array(ohe_sequences), np.array(ohe_predictions))
                ohe_sequences = []
                ohe_predictions = []


# Build model structure
def LSTM_model():
    # Input shape should be sequence length by character length
    # So in this case, it'll be (100, 49)
    # Since we're predicting the next character, the dense layer
    # will output a probability distribution (softmax) over all the possible
    # characters within the text
    # Categorical cross entropy for loss because the predictions are one hot encoded
    model = Sequential()
    model.add(LSTM(128, input_shape=(MAXLEN, len(chars))))
    model.add(Dense(len(chars), activation="softmax"))
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy")

    model.summary()
    return model


# Tune-able parameters
MAXLEN = 100
STEPS = 10
BATCH_SIZE = 32
MODEL_PATH = "bible_generative_model.h5"
EPOCHS = 10
STEPS_PER_EPOCH = (len(text) - MAXLEN - STEPS) // STEPS  # This is determined by
# the number of sequences
# generated
CALLBACKS = [
    ModelCheckpoint(MODEL_PATH)
]

data_generator = sequence_generator(MAXLEN, STEPS, BATCH_SIZE)

# If you're training new model, then assign this as the model
# model = LSTM_model()

# If you're loading model, use this
model = load_model(MODEL_PATH)

# model.fit(data_generator,
#           epochs=EPOCHS,
#           steps_per_epoch=STEPS_PER_EPOCH,
#           callbacks=CALLBACKS)


# One hot encode sentences to feed into our model for prediction
# This will also pad 0 vectors in the front if the length of the sentence
# contains less than MAXLEN characters
def sentence_ohe(sentence):
    ohe_sequence = np.zeros((MAXLEN - len(sentence), len(chars)))
    for c in sentence.lower():
        ohe_char = np.zeros(len(chars))
        ohe_char[char_indices.get(c)] = 1
        ohe_sequence = np.append(ohe_sequence, [ohe_char], axis=0)
    return np.array(ohe_sequence)


TEST_SENTENCE = full_bible_text[100001:100001 + MAXLEN]    # This can be anything,
                                                           # but we'll use it on the next
                                                           # set of text
CHARACTERS_TO_GENERATE = 1000
TEMPERATURE = 1

test_ohe = sentence_ohe(TEST_SENTENCE)
sys.stdout.write(TEST_SENTENCE)                 # Using sys.stdout here instead
                                                # of print because it'll write
                                                # to the same line as we generate
                                                # more characters

# Generate CHARACTERS_TO_GENERATE amount of text after the current TEST_SENTENCE
# Pop the first character after prediction and append the prediction to the
# end of the sequence and continue to feed that into the model for more predictions
for i in range(CHARACTERS_TO_GENERATE):
    dist = model.predict(test_ohe.reshape(1, MAXLEN, len(chars)))
    char_index = ml_utility.sample(dist, TEMPERATURE)
    char_predict = indices_char[char_index]
    sys.stdout.write(char_predict)

    ohe_char = np.zeros(len(chars))
    ohe_char[char_index] = 1
    test_ohe = np.delete(test_ohe, 0, axis=0)
    test_ohe = np.append(test_ohe, [ohe_char], axis=0)