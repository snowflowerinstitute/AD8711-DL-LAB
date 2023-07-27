import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

tf.random.set_seed(69)

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

model = Sequential([
    Dense(4, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=2000, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Loss: {loss:.2f}, Accuracy: {accuracy:.2f}")

y_pred = (model.predict(X, verbose=0) > 0.5).astype("int32")
print("Input\tOutput")
for i in range(len(X)):
    print(f"{X[i]}\t{y_pred[i][0]}")
