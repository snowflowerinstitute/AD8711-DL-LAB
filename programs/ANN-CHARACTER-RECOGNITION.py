import tensorflow as tf
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 784) / 255.
X_test = X_test.reshape(-1, 784) / 255.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, input_shape=(784,), activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

y_pred = model.predict(X_test[:5], verbose=0)
print(f"Predicted: {tf.argmax(y_pred, axis=1)}")
print(f"Actual: {tf.argmax(y_test, axis=1)[:5]}")
