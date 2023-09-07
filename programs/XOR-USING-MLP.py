import tensorflow as tf

tf.random.set_seed(69)

X = tf.convert_to_tensor([[0, 0], [0, 1], [1, 0], [1, 1]], )
y = tf.one_hot([0, 1, 1, 0], depth=2, )

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_dim=2, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, epochs=300, verbose=0)

loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"Loss: {loss:.2f}, Accuracy: {accuracy:.2f}")

y_pred = model.predict(X, verbose=0)
y_pred = tf.argmax(y_pred, axis=1)
print("X", "Predicted", sep='\t\t')
for i, p in enumerate(y_pred):
    print(f"{X[i]}", f"{p}", sep='\t\t')
