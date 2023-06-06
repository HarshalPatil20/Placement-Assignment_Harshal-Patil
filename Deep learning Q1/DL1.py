import tensorflow as tf
from tensorflow.keras import layers

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Model 1: Simple CNN
model1 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Model 2: CNN with Dropout
model2 = tf.keras.Sequential([
    layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Model 3: Deeper CNN
model3 = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile models
model1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train models
model1.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
model2.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
model3.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate models
accuracy1 = model1.evaluate(x_test, y_test)[1]
accuracy2 = model2.evaluate(x_test, y_test)[1]
accuracy3 = model3.evaluate(x_test, y_test)[1]

# Print comparison table
print("Model\t\t\tAccuracy")
print("--------------------------------")
print("Simple CNN\t\t{:.2f}%".format(accuracy1 * 100))
print("CNN with Dropout\t{:.2f}%".format(accuracy2 * 100))
print("Deeper CNN\t\t{:.2f}%".format(accuracy3 * 100))
