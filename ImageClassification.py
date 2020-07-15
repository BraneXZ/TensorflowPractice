from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Dense, AveragePooling2D, Flatten
from tensorflow.keras.datasets import cifar10

import numpy as np

import matplotlib.pyplot as plt

# Loading data and preprocessing by setting values between 0 and 1
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

EPOCHS = 1

def conv2d_model():
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=x_train.shape[1:], activation="relu"))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(Conv2D(64, 3, strides=2, activation="relu"))
    model.add(Conv2D(128, 3, activation="relu"))
    model.add(AveragePooling2D())
    model.add(Dropout(.3))
    model.add(Flatten())

    model.add(Dense(len(np.unique(y_train)), activation="softmax"))

    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])

    model.summary()

    return model

model = conv2d_model()
history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=EPOCHS,
                    validation_split=0.2)


plt.figure(1)
plot_acc(history.history["acc"], history.history["val_acc"], epochs)
plt.figure(2)
plot_loss(history.history["loss"], history.history["val_loss"], epochs)
plt.show()