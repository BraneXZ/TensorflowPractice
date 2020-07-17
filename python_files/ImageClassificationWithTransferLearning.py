from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np

from python_files import ml_utility

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data to between 0 and 1 since pixel values are 0-255
x_train = x_train / 255
x_test = x_test / 255

# Using VGG16 here, can use any pre-train models
pretrain_model = VGG16(include_top=False, weights="imagenet", input_shape=x_train.shape[1:])
pretrain_model.trainable = False


# Create model using pre-trained model
def transfer_model():
    model = Sequential()

    model.add(pretrain_model)
    model.add(Flatten())
    model.add(Dense(512, activation="softmax"))
    model.add(Dense(len(np.unique(y_train)), activation="softmax"))

    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])

    model.summary()

    return model


# Hyper parameters that you can tune
EPOCHS = 2
CALLBACKS = [
    ModelCheckpoint("../model_weights/transfer_learning_model.h5",
                    save_best_only=True),
    EarlyStopping(monitor="val_loss",
                  patience=2)
]
BATCH_SIZE = 32

model = transfer_model()
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    callbacks=CALLBACKS,
                    validation_split=.2)


# Plot training vs validation loss, accuracy
ml_utility.plot_history(history, EPOCHS, ["acc", "loss"])