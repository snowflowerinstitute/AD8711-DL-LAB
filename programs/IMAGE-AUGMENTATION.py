import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import RandomZoom, RandomRotation
from matplotlib import pyplot as plt

(X, _), (_, _) = mnist.load_data()

X = X / 255.

augment = Sequential([
    RandomZoom(0.2),
    RandomRotation(0.2)
])

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
for i in range(5):
    ax = axes[0, i]
    ax.imshow(X[i], cmap='gray')
    ax.set_title('Original')
    ay = axes[1, i]
    ay.imshow(augment(tf.expand_dims(X[i], 0))[0], cmap='gray')
    ay.set_title('Augmented')

plt.show()
