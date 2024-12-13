import numpy as np

class SigmoidActivation:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self(x)
        return sig * (1 - sig)

class IdentityActivation:
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1

class DenseLayer:
    def __init__(self, input_size, output_size, activation, weight_initializer='random', bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.include_bias = bias
        self.activation_function = self._select_activation(activation)
        self.weights = self._initialize_weights(weight_initializer)

    def _select_activation(self, activation):
        if activation == 'sigmoid':
            return SigmoidActivation()
        elif activation == 'identity':
            return IdentityActivation()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def _initialize_weights(self, initializer):
        if self.include_bias:
            shape = (self.input_size + 1, self.output_size + 1)  # Include bias term
        else:
            shape = (self.input_size + 1, self.output_size)
        
        if initializer == 'random':
            return np.random.randn(*shape)
        elif initializer == 'zeros':
            return np.zeros(shape)
        else:
            raise ValueError(f"Unknown weight initialization method: {initializer}")

    def __str__(self):
        return str(self.weights)

    def forward(self, input_data):
        return self.activation_function(np.dot(input_data, self.weights))

    def backward(self, activations, deltas):
        error = np.dot(deltas[-1], self.weights.T) * self.activation_function.derivative(activations)
        return error

    def update_weights(self, learning_rate, activations, deltas):
        gradient = np.dot(activations.T, deltas)
        self.weights -= learning_rate * gradient
        return gradient

class MLP:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input_data):
        input_data = np.append(1, input_data)  # Adding bias input
        activations = [np.atleast_2d(input_data)]  # Store activations for each layer

        for layer in self.layers:
            output = layer.forward(activations[-1])
            activations.append(output)

        return float(activations[-1]), activations

    def train(self, activations, target_output, learning_rate=0.1, verbose=False):
        deltas = [activations[-1] - target_output]  # Calculate the output layer error

        for i in range(len(activations) - 2, 0, -1):
            error = self.layers[i].backward(activations[i], deltas)
            deltas.append(error)

        deltas.reverse()

        for i, layer in enumerate(self.layers):
            grad = layer.update_weights(learning_rate, activations[i], deltas[i])
            if verbose:
                print(f"Layer {i+1} weight gradients: \n{grad}")
