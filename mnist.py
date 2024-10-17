import numpy as np
from sklearn.datasets import fetch_openml

from layer import ActivationLayer, ConnectedLayer
from network import Network
from utils import to_categorical

# Load MNIST dataset from scikit-learn
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print("Dataset loaded.")

# Prepare training data
x_train = mnist.data[:40000]
y_train = mnist.target[:40000]

# Reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = to_categorical(y_train)

# Network
net = Network()
net.add(ConnectedLayer(28*28, 100))
net.add(ActivationLayer())
net.add(ConnectedLayer(100, 50))
net.add(ActivationLayer())
net.add(ConnectedLayer(50, 10))
net.add(ActivationLayer())

# Train on all 40000 samples
losses = net.fit(x_train, y_train, epochs=100, learning_rate=0.01)

x_check = mnist.data
counter = 0

# Test on 20000 samples (from 40000 to 60000 so we don't test on the training data)
print("")
for k in range(20000):
  x = x_check[k + 40000]
  x = x.reshape(1, 1, 28*28)
  x = x.astype('float32')
  x = np.float32(x / 255)
  prediction = net.predict(x)
  result = np.argmax(prediction)
  symbol = "✓" if result == int(mnist.target[k + 40000]) else "✗"
  print(f"{symbol} Prediction: {result}, expected: {int(mnist.target[k + 40000])} ({counter * 100 / (k + 1)}% accuracy)", end='\r')
  counter += 1 if result == int(mnist.target[k + 40000]) else 0
print("")

print(f"{counter * 100 / 20000}% accuracy")

net.save_weights('test/tensor_func.npz')
