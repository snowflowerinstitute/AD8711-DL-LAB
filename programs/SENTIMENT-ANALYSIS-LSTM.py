from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import pad_sequences

import numpy as np

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

max_len = 100
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential([
    Embedding(10000, 192),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)

score, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
print(f"Test score: {score:.2f} - Test accuracy: {acc:.2f}")

word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}


def decode_review(review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in review])


random_indices = np.random.randint(0, len(x_test), 5)
x_sample = x_test[random_indices]
y_sample = y_test[random_indices]
y_pred = model.predict(x_sample, batch_size=5, verbose=0)

for i in range(len(x_sample)):
    print(f"Review: {decode_review(x_sample[i])[0:30]}...")
    print(f"Sentiment: {'Positive' if y_sample[i] == 1 else 'Negative'}")
    print(f"Predicted sentiment: {'Positive' if y_pred[i] > 0.5 else 'Negative'}")
    confidence = y_pred[i] if y_pred[i] > 0.5 else 1 - y_pred[i]
    print(f"Confidence: {confidence[0]:.2f}")
    print('-' * 50)
