import os
import csv
import numpy as np
from perceptron import Perceptron, VotedPerceptron, AveragedPerceptron

# Set the seed for random number generation
np.random.seed(33)

# Define file paths and output directory
data_path = "../data/bank-note/"
output_folder = './out/'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Helper function to load data
def load_data(file_name):
    file_data = np.genfromtxt(file_name, delimiter=',')
    features = file_data[:, :-1]  # All columns except the last
    labels = np.where(file_data[:, -1] == 0, -1, 1)  # Convert labels: 0 -> -1, 1 stays 1
    return features, labels

# Load training and test datasets
train_features, train_labels = load_data(data_path + 'train.csv')
test_features, test_labels = load_data(data_path + 'test.csv')

# Function to compute and print model performance
def print_performance(model, train_data, train_labels, test_data, test_labels):
    train_accuracy = np.mean(train_labels == model.predict(train_data))
    test_accuracy = np.mean(test_labels == model.predict(test_data))
    return train_accuracy, test_accuracy

# Standard Perceptron Classifier
print("==== Standard Perceptron ====")
perceptron_classifier = Perceptron(train_features, train_labels, r=0.1)
train_acc, test_acc = print_performance(perceptron_classifier, train_features, train_labels, test_features, test_labels)
print(f"Trained weights: {perceptron_classifier.weights}")
print(f"Training accuracy: {train_acc}")
print(f"Testing accuracy: {test_acc}")

# Voted Perceptron Classifier
print("==== Voted Perceptron ====")
voted_classifier = VotedPerceptron(train_features, train_labels, r=0.1)
train_acc, test_acc = print_performance(voted_classifier, train_features, train_labels, test_features, test_labels)
print(f"Learned weights and counts: {voted_classifier.votes}")

# Saving Voted Perceptron Weights to CSV
weights_file_path = os.path.join(output_folder, 'voted_perceptron_weights.csv')
with open(weights_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['b', 'x1', 'x2', 'x3', 'x4', 'Cm'])  # Column headers
    for weight, count in voted_classifier.votes:
        writer.writerow(np.append(weight, count))

print(f"Training accuracy: {train_acc}")
print(f"Testing accuracy: {test_acc}")

# Averaged Perceptron Classifier
print("==== Averaged Perceptron ====")
averaged_classifier = AveragedPerceptron(train_features, train_labels, r=0.1)
train_acc, test_acc = print_performance(averaged_classifier, train_features, train_labels, test_features, test_labels)
print(f"Trained weights: {averaged_classifier.weights}")
print(f"Training accuracy: {train_acc}")
print(f"Testing accuracy: {test_acc}")
