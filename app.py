from flask import Flask, request, jsonify, render_template
import numpy as np
from network import Network
from layer import ConnectedLayer, ActivationLayer

app = Flask(__name__)

# Load your trained network
net = Network()
net.add(ConnectedLayer(28*28, 392))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(ConnectedLayer(392, 196))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(ConnectedLayer(196, 10))
net.add(ActivationLayer(tanh, tanh_prime))

# Load the weights (you need to save these after training)
net.load_weights('mnist_weights.npz')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.json['image']
    image_array = np.array(image_data).reshape(1, 1, 28*28)
    image_array = image_array.astype('int16')

    prediction = net.predict(image_array)
    result = np.argmax(prediction)

    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(debug=True)
