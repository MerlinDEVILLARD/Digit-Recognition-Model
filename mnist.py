import numpy as np
from sklearn.datasets import fetch_openml

from layer import ActivationLayer, ConnectedLayer
from network import Network
import matplotlib.pyplot as plt

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

# def to_categorical(y, num_classes=None, dtype='int16'):
#   y = np.array(y, dtype='int')
#   input_shape = y.shape
#   if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#     input_shape = tuple(input_shape[:-1])
#   y = y.ravel()
#   if not num_classes:
#     num_classes = np.max(y) + 1
#   n = y.shape[0]
#   categorical = np.zeros((n, num_classes), dtype=dtype)
#   categorical[np.arange(n), y] = 1
#   output_shape = input_shape + (num_classes,)
#   categorical = np.reshape(categorical, output_shape)
#   return categorical

def to_categorical(y, num_classes=10):
    return np.eye(num_classes)[y.astype(int)]


y_train = to_categorical(y_train)

# Network
net = Network()
net.add(ConnectedLayer(28*28, 100))
net.add(ActivationLayer())
net.add(ConnectedLayer(100, 50))
net.add(ActivationLayer())
net.add(ConnectedLayer(50, 10))
net.add(ActivationLayer())

# Train on all 60000 samples
losses = net.fit(x_train, y_train, epochs=100, learning_rate=0.01)

x_check = mnist.data
counter = 0

for k in range(20000):
    x = x_check[k + 40000]
    x = x.reshape(1, 1, 28*28)
    x = x.astype('float32')
    x = np.float32(x / 255)
    prediction = net.predict(x)
    result = np.argmax(prediction)
    print(f"Prediction: {result}")
    print(f"Actual: {mnist.target[k + 40000]}")
    counter += 1 if result == int(mnist.target[k + 40000]) else 0
    print("")

print(f"{counter * 100 / 20000}% accuracy")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(losses) + 1), losses)
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_vs_epochs.png')

net.save_weights('mnist_weightszbeub.npz')
