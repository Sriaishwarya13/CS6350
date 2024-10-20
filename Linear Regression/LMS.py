import sys
import numpy as np
from numpy import linalg as linalg

class LinearRegression:
    def __init__(self, learning_rate=1.0, convergence_threshold=0.0001):
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.weights = None

    def fit_stochastic(self, data, targets):
        """Fits the model using Stochastic Gradient Descent."""
        self.weights = np.zeros((101, 7), dtype="float")
        cost_log = "Learning Rate: " + str(self.learning_rate) + ":      "
        prev_cost = float('inf')

        for epoch in range(100):
            shuffled_data, shuffled_targets = self._shuffle_data(data, targets)

            for i in range(53):
                h = shuffled_data[i] @ self.weights[epoch].T
                error = shuffled_targets[i] - h

                for j in range(7):
                    self.weights[epoch + 1, j] = self.weights[epoch, j] + self.learning_rate * error * shuffled_data[i, j]

                curr_cost = self._compute_cost(shuffled_data, shuffled_targets, self.weights[epoch + 1])
                cost_log += str(curr_cost) + ","

            if abs(curr_cost - prev_cost) < self.convergence_threshold:
                print(cost_log)
                return self.weights[epoch + 1]
            prev_cost = curr_cost

        print(cost_log)
        self.learning_rate *= 0.5

    def fit_batch(self, data, targets):
        """Fits the model using Batch Gradient Descent."""
        self.weights = np.zeros((10001, 7), dtype="float")
        cost_log = "Learning Rate: " + str(self.learning_rate) + ":      "
        prev_iteration = -1

        for epoch in range(10000):
            gradient = self._compute_gradient(data, targets, self.weights[epoch])
            self.weights[epoch + 1] = self.weights[epoch] - self.learning_rate * gradient

            cost_log += str(self._compute_cost(data, targets, self.weights[epoch])) + ","
            if linalg.norm(self.weights[epoch + 1] - self.weights[epoch]) < self.convergence_threshold:
                print(cost_log)
                return self.weights[epoch + 1]

        print(cost_log)
        self.learning_rate *= 0.5

    def _compute_gradient(self, data, targets, weights):
        """Calculates the gradient for the Least Mean Squares cost function."""
        predictions = data @ weights.T
        errors = targets - predictions
        gradient = -data.T @ errors / len(targets)
        return gradient

    def _compute_cost(self, data, targets, weights):
        """Calculates the Least Mean Squares cost."""
        predictions = data @ weights.T
        errors = targets - predictions
        return 0.5 * np.sum(errors ** 2)

    def _shuffle_data(self, data, targets):
        """Shuffles the dataset."""
        combined = np.c_[data, targets]
        np.random.shuffle(combined)
        return combined[:, :-1], combined[:, -1]

def load_data(filepath, num_samples):
    """Loads data from a CSV file."""
    data = np.empty((num_samples, 7), dtype="float")
    targets = np.empty((num_samples, 1), dtype="float")

    with open(filepath, 'r') as file:
        for i, line in enumerate(file):
            elements = line.strip().split(',')
            data[i] = [float(elements[j]) for j in range(7)]
            targets[i] = float(elements[7])
    return np.asmatrix(data), np.asmatrix(targets)

if __name__ == '__main__':
    train_data, train_targets = load_data("./concrete/train.csv", 53)
    test_data, test_targets = load_data("./concrete/test.csv", 50)

    model = LinearRegression()

    if sys.argv[1] == "bgd":
        final_weights = model.fit_batch(train_data, train_targets)
        print("Final Weights: ", final_weights)
        print("Test Cost: ", model._compute_cost(test_data, test_targets, final_weights))
    elif sys.argv[1] == "sgd":
        final_weights = model.fit_stochastic(train_data, train_targets)
        print("Final Weights: ", final_weights)
        print("Test Cost: ", model._compute_cost(test_data, test_targets, final_weights))

