from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255

vgg = VGG16(include_top=False, weights="imagenet", input_shape=x_train.shape[1:])
vgg.trainable = False

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(512, activation="softmax"))
model.add(Dense(len(np.unique(y_train)), activation="softmax"))

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["acc"])

model.summary()

epochs = 10
callbacks = [
    ModelCheckpoint("transfer_learning_model.h5",
                    save_best_only=True),
    EarlyStopping(monitor="val_loss",
                  patience=2)
]
batch_size = 32

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=.2)

training_acc = history.history["acc"]
val_acc = history.history["val_acc"]
training_loss = history.history["loss"]
val_loss = history.history["val_loss"]


def plot_acc(acc, val_acc, epochs):
    plt.plot(range(1, epochs+1), acc, label="Training Accuracy")
    plt.plot(range(1, epochs+1), val_acc, label="Validation Accuracy")

    plt.legend()
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")


def plot_loss(loss, val_loss, epochs):
    plt.plot(range(1, epochs+1), loss, label="Training Loss")
    plt.plot(range(1, epochs+1), val_loss, label="Validation Loss")

    plt.legend()
    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


plt.figure(1)
plot_acc(training_acc, val_acc, epochs)
plt.figure(2)
plot_loss(training_loss, val_loss, epochs)
plt.show()





