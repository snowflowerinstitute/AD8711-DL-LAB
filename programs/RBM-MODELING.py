from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM

(X, _), (_, _) = mnist.load_data()
X = X.reshape(-1, 784) / 255.

rbm = BernoulliRBM(n_components=128, learning_rate=0.01, batch_size=10, n_iter=10, verbose=1)
rbm.fit(X)

plt.figure(figsize=(10, 10))
for i, comp in enumerate(rbm.components_):
    plt.subplot(16, 16, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap='gray')
    plt.xticks(())
    plt.yticks(())

plt.suptitle('Components extracted by RBM', fontsize=16)
plt.show()
