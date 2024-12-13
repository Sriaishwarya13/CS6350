import neural
import numpy as np

print("Start test")

model = nn.NeuralNetwork([
    nn.FCLayer(input_dim=4, output_dim=5, activation='sigmoid', weight_init_method='random'),  # input layer
    nn.FCLayer(input_dim=5, output_dim=5, activation='sigmoid', weight_init_method='random'),  # hidden layer
    nn.FCLayer(input_dim=5, output_dim=1, activation='identity', weight_init_method='random', use_bias=False)  # output layer
])

input_data = np.array([-1.7582, 2.7397, -2.5323, -2.234])
desired_output = 1

predicted_output, activations = model.forward(input_data)
model.backward(activations, desired_output, show_details=True)
