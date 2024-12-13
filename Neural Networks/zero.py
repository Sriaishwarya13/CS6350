import nn
import numpy as np
import os
import matplotlib.pyplot as plt

os.makedirs("./out/", exist_ok=True)

data_path = "../data/bank-note/"

def loss_fn(pred, true):
    return 0.5 * (pred - true) ** 2

def load_data(filepath):
    features, labels = [], []
    with open(filepath, "r") as file:
        for line in file:
            data = list(map(np.float128, line.strip().split(",")))
            features.append(data[:-1])
            labels.append(data[-1])
    return np.array(features), np.array(labels)

X_train, y_train = load_data(data_path + "train.csv")
X_test, y_test = load_data(data_path + "test.csv")

def train_network(epochs, model, X_train, y_train, learning_rate=0.5, decay=1):
    epoch_losses = []
    for epoch in range(epochs):
        batch_losses = []
        idx = np.random.permutation(len(X_train))
        for i in idx:
            output, activations = model.forward(X_train[i])
            batch_losses.append(loss_fn(output, y_train[i]))

            lr = learning_rate / (1 + (learning_rate / decay) * epoch)
            model.backward(activations, y_train[i], lr)

        print(f"Epoch {epoch+1} loss: {np.mean(batch_losses):.6f}")
        epoch_losses.append(np.mean(batch_losses))

    return epoch_losses

def evaluate(model, X_test, y_test):
    test_losses = []
    for i in range(len(X_test)):
        pred, _ = model.forward(X_test[i])
        test_losses.append(loss_fn(pred, y_test[i]))

    print(f"Test loss: {np.mean(test_losses):.6f}")
    return np.mean(test_losses)

def experiment(input_size, hidden_size, output_size, lr, width, epochs=35):
    print(f"{width}-unit network with zero initialization:\n-----------------------------------")
    network = nn.NeuralNetwork([
        nn.FCLayer(input_dim=input_size, output_dim=hidden_size, activation='sigmoid', weight_init_method='zeroes'),
        nn.FCLayer(input_dim=hidden_size, output_dim=hidden_size, activation='sigmoid', weight_init_method='zeroes'),
        nn.FCLayer(input_dim=hidden_size, output_dim=output_size, activation='identity', weight_init_method='zeroes', use_bias=False)
    ])

    training_losses = train_network(epochs, network, X_train, y_train, learning_rate=lr, decay=1)
    plt.figure()
    plt.plot(training_losses)
    plt.title(f"Width = {width}, Zero Initialization")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.savefig(f"./out/zeroes_w{width}.png")

    test_loss = evaluate(network, X_test, y_test)
    return test_loss

experiment(4, 5, 1, lr=0.5, width=5)
experiment(4, 10, 1, lr=0.5, width=10)
experiment(4, 25, 1, lr=0.05, width=25)
experiment(4, 50, 1, lr=0.1, width=50)
experiment(4, 100, 1, lr=0.01, width=100)

plt.show()
