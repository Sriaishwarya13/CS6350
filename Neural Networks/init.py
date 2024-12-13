import neural
import numpy as np
import os
import matplotlib.pyplot as plt

output_dir = "./out/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset_path = "../data/bank-note/"

def compute_loss(prediction, target):
    return 0.5 * np.square(prediction - target)

def load_dataset(file_path):
    inputs, targets = [], []
    with open(file_path, "r") as file:
        for line in file:
            data = line.strip().split(",")
            data = list(map(np.float128, data))
            inputs.append(data[:-1])
            targets.append(data[-1])
    return np.array(inputs), np.array(targets)

train_features, train_labels = load_dataset(dataset_path + "train.csv")
test_features, test_labels = load_dataset(dataset_path + "test.csv")

def train_network(epochs, model, train_features, train_labels, initial_lr=0.5, decay_rate=1):
    epoch_losses = []

    for epoch in range(epochs):
        batch_losses = []
        shuffled_indices = np.random.permutation(len(train_features))
        for index in shuffled_indices:
            prediction, activations = model.forward(train_features[index])
            batch_losses.append(compute_loss(prediction, train_labels[index]))

            learning_rate = initial_lr / (1 + (initial_lr / decay_rate) * epoch)
            model.backward(activations, train_labels[index], learning_rate)
        
        avg_batch_loss = np.mean(batch_losses)
        print(f"Epoch {epoch + 1} average training loss: {avg_batch_loss:.6f}")
        epoch_losses.append(avg_batch_loss)

    return epoch_losses

def evaluate_network(model, test_features, test_labels):
    test_losses = []
    for i in range(len(test_features)):
        prediction, _ = model.forward(test_features[i])
        test_losses.append(compute_loss(prediction, test_labels[i]))
    
    avg_test_loss = np.mean(test_losses)
    print(f"Test loss: {avg_test_loss:.6f}\n")
    return avg_test_loss

print("Training network with 5 units per layer")
network_5 = nn.NeuralNetwork([
    nn.FCLayer(4, 5, 'sigmoid', 'random'),
    nn.FCLayer(5, 5, 'sigmoid', 'random'),
    nn.FCLayer(5, 1, 'identity', 'random', include_bias=False)
])

training_losses_5 = train_network(35, network_5, train_features, train_labels, initial_lr=0.5, decay_rate=1)
fig, ax = plt.subplots()
ax.plot(training_losses_5)
ax.set_title("Width = 5, Random Initialization")
ax.set_xlabel("Epochs")
ax.set_ylabel("Squared Error")
plt.savefig(output_dir + "random_w5.png")
evaluate_network(network_5, test_features, test_labels)

print("Training network with 10 units per layer")
network_10 = nn.NeuralNetwork([
    nn.FCLayer(4, 10, 'sigmoid', 'random'),
    nn.FCLayer(10, 10, 'sigmoid', 'random'),
    nn.FCLayer(10, 1, 'identity', 'random', include_bias=False)
])

training_losses_10 = train_network(35, network_10, train_features, train_labels, initial_lr=0.5, decay_rate=1)
fig, ax = plt.subplots()
ax.plot(training_losses_10)
ax.set_title("Width = 10, Random Initialization")
ax.set_xlabel("Epochs")
ax.set_ylabel("Squared Error")
plt.savefig(output_dir + "random_w10.png")
evaluate_network(network_10, test_features, test_labels)

print("Training network with 25 units per layer")
network_25 = nn.NeuralNetwork([
    nn.FCLayer(4, 25, 'sigmoid', 'random'),
    nn.FCLayer(25, 25, 'sigmoid', 'random'),
    nn.FCLayer(25, 1, 'identity', 'random', include_bias=False)
])

training_losses_25 = train_network(35, network_25, train_features, train_labels, initial_lr=0.05, decay_rate=1)
fig, ax = plt.subplots()
ax.plot(training_losses_25)
ax.set_title("Width = 25, Random Initialization")
ax.set_xlabel("Epochs")
ax.set_ylabel("Squared Error")
plt.savefig(output_dir + "random_w25.png")
evaluate_network(network_25, test_features, test_labels)

print("Training network with 50 units per layer")
network_50 = nn.NeuralNetwork([
    nn.FCLayer(4, 50, 'sigmoid', 'random'),
    nn.FCLayer(50, 50, 'sigmoid', 'random'),
    nn.FCLayer(50, 1, 'identity', 'random', include_bias=False)
])

training_losses_50 = train_network(35, network_50, train_features, train_labels, initial_lr=0.1, decay_rate=1)
fig, ax = plt.subplots()
ax.plot(training_losses_50)
ax.set_title("Width = 50, Random Initialization")
ax.set_xlabel("Epochs")
ax.set_ylabel("Squared Error")
plt.savefig(output_dir + "random_w50.png")
evaluate_network(network_50, test_features, test_labels)

print("Training network with 100 units per layer")
network_100 = nn.NeuralNetwork([
    nn.FCLayer(4, 100, 'sigmoid', 'random'),
    nn.FCLayer(100, 100, 'sigmoid', 'random'),
    nn.FCLayer(100, 1, 'identity', 'random', include_bias=False)
])

training_losses_100 = train_network(35, network_100, train_features, train_labels, initial_lr=0.01, decay_rate=2)
fig, ax = plt.subplots()
ax.plot(training_losses_100)
ax.set_title("Width = 100, Random Initialization")
ax.set_xlabel("Epochs")
ax.set_ylabel("Squared Error")
plt.savefig(output_dir + "random_w100.png")
evaluate_network(network_100, test_features, test_labels)

plt.show()
