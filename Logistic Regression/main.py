import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_data():
    train_data = pd.read_csv('classification/train.csv', header=None)
    test_data = pd.read_csv('classification/test.csv', header=None)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradients(X, y, w, v=0, is_MAP=False):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, w))
    error = predictions - y
    gradient = np.dot(X.T, error) / m
    if is_MAP:
        gradient += (w / v)
    return gradient

def logistic_regression(X_train, y_train, X_test, y_test, learning_rate_schedule, epochs=100, v=0, is_MAP=False):
    w = np.zeros(X_train.shape[1])
    errors_train = []
    errors_test = []
    
    for epoch in range(epochs):
        perm = np.random.permutation(X_train.shape[0])
        X_train = X_train[perm]
        y_train = y_train[perm]
        gamma = learning_rate_schedule(epoch)
        
        for i in range(X_train.shape[0]):
            gradient = compute_gradients(X_train[i:i+1], y_train[i:i+1], w, v, is_MAP)
            w -= gamma * gradient
        
        train_error = compute_error(X_train, y_train, w)
        test_error = compute_error(X_test, y_test, w)
        
        errors_train.append(train_error)
        errors_test.append(test_error)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
    
    return w, errors_train, errors_test

def compute_error(X, y, w):
    predictions = sigmoid(np.dot(X, w)) > 0.5
    return np.mean(predictions != y)

def learning_rate_schedule(t, gamma_0=0.1, d=10):
    return gamma_0 / (1 + gamma_0 * d * t)

def run_experiment():
    X_train, y_train, X_test, y_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)
    
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    for v in variances:
        print(f"Running MAP estimation with prior variance v={v}")
        w_map, train_errors_map, test_errors_map = logistic_regression(X_train, y_train, X_test, y_test, learning_rate_schedule, v=v, is_MAP=True)
        
        plt.plot(train_errors_map, label=f'Train (v={v})')
        plt.plot(test_errors_map, label=f'Test (v={v})')
    
    plt.legend()
    plt.title("MAP Estimation Errors")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()
    
    print("Running ML estimation (no prior)")
    w_ml, train_errors_ml, test_errors_ml = logistic_regression(X_train, y_train, X_test, y_test, learning_rate_schedule, v=0, is_MAP=False)
    
    plt.plot(train_errors_ml, label='Train (ML)')
    plt.plot(test_errors_ml, label='Test (ML)')
    
    plt.legend()
    plt.title("ML Estimation Errors")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

if __name__ == "__main__":
    run_experiment()
