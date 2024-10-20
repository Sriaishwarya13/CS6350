import math
import random
import sys
import numpy as np
sys.path.append('../DecisionTree')
import ID3

NUM_ITERATIONS = 1000
NUM_SAMPLES = 4999
trees = []
weights = []
attribute_sizes = [2, 4, 6]

# Main execution flow
def main():
    ID3.data_type = "bank"
    algorithm = sys.argv[1]

    if algorithm == "ada":
        ID3.max_depth = 2
        predictions = np.zeros((NUM_ITERATIONS, NUM_SAMPLES))
        print("Running Adaboost")
        run_adaboost(predictions)
    elif algorithm == "bag":
        predictions = np.zeros(NUM_SAMPLES)
        print("Running Bagged Trees")
        run_bagged_trees(predictions)
    elif algorithm == "forest":
        print("Running Random Forest")
        run_random_forest()

def run_adaboost(predictions):
    """Executes the Adaboost algorithm."""
    train_labels = np.array(ID3.train_data[-1])
    test_labels = np.array(ID3.test_data[-1])
    train_errors, test_errors = "", ""

    for i in range(NUM_ITERATIONS):
        hypothesis, current_weights = adaboost_iteration(i, train_labels, predictions)
        train_errors += f"{compute_error(train_labels, hypothesis)},"
        test_hypothesis = test_adaboost(current_weights, predictions)
        test_errors += f"{compute_error(test_labels, test_hypothesis)},"

    print("TRAINING ERRORS:\n", train_errors)
    print("TESTING ERRORS:\n", test_errors)
    display_stump_errors()

def adaboost_iteration(i, labels, predictions):
    """Trains a decision stump and updates weights."""
    stump = ID3.train(ID3.train_data, i)
    trees.append(stump)
    predictions[i] = predict_with_tree(ID3.train_data, stump.root)

    error_rate = calculate_error(labels, i)
    weights.append(calculate_weight(error_rate))

    if i < NUM_ITERATIONS - 1:
        update_example_weights(labels, i)

    final_hypothesis = aggregate_hypotheses(weights, predictions)
    return final_hypothesis, weights

def test_adaboost(weights, predictions):
    """Generates predictions on test data using trained trees."""
    for i in range(len(trees)):
        predictions[i] = predict_with_tree(ID3.test_data, trees[i].root)
    return aggregate_hypotheses(weights, predictions)

def calculate_error(labels, i):
    """Calculates the error for predictions."""
    return 0.5 - (0.5 * np.sum(ID3.example_weights[i] * labels * predictions[i]))

def calculate_weight(error):
    """Computes the weight for a given error."""
    return 0.5 * math.log((1.0 - error) / error)

def update_example_weights(labels, i):
    """Updates weights based on current predictions."""
    ID3.example_weights[i + 1] = ID3.example_weights[i] * np.exp(-weights[i] * labels * predictions[i])
    total_weight = np.sum(ID3.example_weights[i + 1])
    ID3.example_weights[i + 1] /= total_weight

def aggregate_hypotheses(weights, predictions):
    """Combines predictions weighted by their corresponding weights."""
    combined = np.sum([weights[j] * predictions[j] for j in range(len(weights))], axis=0)
    return np.sign(combined)

# Bagged Trees Implementation
def run_bagged_trees(predictions):
    """Executes the bagged decision tree algorithm."""
    train_errors, test_errors = "", ""

    for i in range(NUM_ITERATIONS):
        train_err, predictions = bagged_iteration(i, True, predictions)
        train_errors += f"{train_err},"
        test_err, _ = bagged_iteration(i, False, predictions)
        test_errors += f"{test_err},"

    print("TRAINING ERRORS:\n", train_errors)
    print("TESTING ERRORS:\n", test_errors)

def bagged_iteration(i, is_training, predictions):
    """Performs one iteration of the bagged decision tree algorithm."""
    sampled_data = sample_with_replacement(ID3.train_data)
    if is_training:
        trees.append(ID3.train(sampled_data, i, attribute_subset_size))
    
    predictions = compute_bagged_predictions(sampled_data, trees[i].root, predictions)
    final_predictions = finalize_bagged_predictions(predictions)

    return compute_error(np.array(sampled_data[-1], dtype=int), final_predictions), predictions

def sample_with_replacement(data):
    """Generates samples with replacement from the dataset."""
    indices = [random.randint(0, len(data[-1]) - 1) for _ in range(NUM_SAMPLES)]
    return [[data[j][index] for index in indices] for j in range(len(data))]

def finalize_bagged_predictions(predictions):
    """Finalizes predictions, converting zeros to -1 or 1 as needed."""
    final_hypotheses = np.sign(predictions)
    for i in range(predictions.size):
        if predictions[i] == 0:
            final_hypotheses[i] = 1 if i % 2 == 0 else -1
    return final_hypotheses

# Random Forest Implementation
def run_random_forest():
    """Executes the random forest algorithm across different attribute sizes."""
    for attribute_subset_size in attribute_sizes:
        trees.clear()
        print("ATTRIBUTE SIZE:", attribute_subset_size)
        run_bagged_trees(np.zeros(NUM_SAMPLES))

# Prediction Functions
def predict_with_tree(data, root):
    """Predicts outcomes using the given tree."""
    return np.array([ID3.predict_example([feature[i] for feature in data], root, False) for i in range(len(data[-1]))])

def compute_error(labels, predictions):
    """Computes the proportion of incorrect predictions."""
    return np.mean(labels != predictions)

def display_stump_errors():
    """Displays errors for each stump."""
    errors = ""
    for i in range(NUM_ITERATIONS):
        errors += f"{ID3.calculate_prediction_error_for_tree(ID3.train_data, trees[i].root)},"
    print("STUMP TRAINING ERRORS:\n", errors)

if __name__ == '__main__':
    main()
