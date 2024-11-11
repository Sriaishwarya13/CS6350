import numpy as np

class Perceptron:
    def __init__(self, X=None, y=None, learning_rate: float = 1e-3, n_epochs: int = 10):
        self.weights = None
        if X is not None and y is not None:
            self.fit(X, y, learning_rate, n_epochs)

    def add_bias_term(self, X):
        return np.c_[np.ones(X.shape[0]), X]  # Add bias column (1's) to feature matrix

    def fit(self, X, y, learning_rate: float = 1e-3, n_epochs: int = 10):
        X_with_bias = self.add_bias_term(X)  # Augment input data with bias term
        self.weights = np.zeros(X_with_bias.shape[1])  # Initialize weights to zeros

        for _ in range(n_epochs):
            permuted_indices = np.random.permutation(len(X_with_bias))  # Shuffle indices
            for i in permuted_indices:
                if y[i] * np.dot(self.weights, X_with_bias[i]) <= 0:  # Misclassification
                    self.weights += learning_rate * y[i] * X_with_bias[i]  # Update weights

    def predict(self, X) -> np.ndarray:
        X_with_bias = self.add_bias_term(X)  # Add bias term to test set
        return np.sign(np.dot(X_with_bias, self.weights))  # Return predictions (-1 or 1)

class VotedPerceptron(Perceptron):
    def __init__(self, X, y, learning_rate: float = 1e-3, n_epochs: int = 10):
        self.votes = []
        self.fit(X, y, learning_rate, n_epochs)

    def fit(self, X, y, learning_rate: float = 1e-3, n_epochs: int = 10):
        X_with_bias = self.add_bias_term(X)  # Add bias column
        current_index = 0
        weight_vectors = [np.zeros(X_with_bias.shape[1])]  # Store weight vectors
        vote_counts = [0]  # Store vote counts for each weight vector

        for _ in range(n_epochs):
            permuted_indices = np.random.permutation(len(X_with_bias))  # Shuffle indices
            for i in permuted_indices:
                if y[i] * np.dot(weight_vectors[current_index], X_with_bias[i]) <= 0:  # Misclassification
                    weight_vectors[current_index] += learning_rate * y[i] * X_with_bias[i]  # Update weight vector
                    weight_vectors.append(weight_vectors[current_index].copy())  # Append new weight
                    vote_counts.append(1)  # Initialize count for new weight vector
                    current_index += 1
                else:
                    vote_counts[current_index] += 1  # Increment count for current weight vector

        self.votes = list(zip(weight_vectors, vote_counts))  # Store all weight vectors and their counts

    def predict(self, X) -> np.ndarray:
        X_with_bias = self.add_bias_term(X)  # Add bias column
        predictions = np.zeros(len(X), dtype=int)  # Initialize predictions array

        for i in range(len(X)):
            total_sum = 0
            for w, count in self.votes:
                total_sum += count * np.sign(np.dot(w, X_with_bias[i]))  # Weighted sum of predictions
            predictions[i] = np.sign(total_sum)  # Final prediction

        return predictions

class AveragedPerceptron(Perceptron):
    def fit(self, X, y, learning_rate: float = 1e-3, n_epochs: int = 10):
        X_with_bias = self.add_bias_term(X)  # Add bias term to features
        self.weights = np.zeros(X_with_bias.shape[1])  # Initialize weights
        cumulative_weights = np.zeros(X_with_bias.shape[1])  # Accumulate weights

        for _ in range(n_epochs):
            permuted_indices = np.random.permutation(len(X_with_bias))  # Shuffle indices
            for i in permuted_indices:
                if y[i] * np.dot(cumulative_weights, X_with_bias[i]) <= 0:  # Misclassification
                    cumulative_weights += learning_rate * y[i] * X_with_bias[i]  # Update cumulative weights
            self.weights += cumulative_weights  # Update the averaged weights after each epoch
