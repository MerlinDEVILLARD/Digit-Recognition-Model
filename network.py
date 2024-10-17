import numpy as np

# Loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        input_data_len = len(input_data)
        result = []

        # run network over all data
        for i in range(input_data_len):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        x_train_len = len(x_train)
        losses = []

        for i in range(epochs):
            err = 0
            for j in range(x_train_len):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # using mse to compute loss
                err += mse(y_train[j], output)

                # using mse_prime to compute loss
                error = mse_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            losses.append(err)
            # calculate average error on all samples
            err /= x_train_len
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def save_weights(self, filename):
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                weights[f'layer_{i}_weights'] = layer.weights
            if hasattr(layer, 'bias'):
                weights[f'layer_{i}_bias'] = layer.bias
        np.savez(filename, **weights)

    def load_weights(self, filename):
        weights = np.load(filename)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                layer.weights = weights[f'layer_{i}_weights']
            if hasattr(layer, 'bias'):
                layer.bias = weights[f'layer_{i}_bias']

