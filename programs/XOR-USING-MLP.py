from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Define the training data
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Define the MLP model
model = Sequential([
    Dense(4, input_dim=2, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=2000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Predict the output for new input values
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("Input\tOutput")
for i in range(len(X_test)):
    print(f"{X_test[i]}\t{y_pred[i][0]}")
