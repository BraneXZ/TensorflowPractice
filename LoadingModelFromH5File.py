from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10

(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test / 255

model_path = "transfer_learning_model.h5"

model = load_model(model_path)


predictions = model.predict(x_test[:5])

for p, a in zip(predictions, y_test[:5]):
    print(f"Prediction: {p} Actual: {a}")
