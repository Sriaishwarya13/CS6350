import numpy as np
import pandas as pd
from scipy import optimize
import math


df_train = pd.read_csv('bank-note/train.csv', header=None)
df_test = pd.read_csv('bank-note/test.csv', header=None)
column = ['var', 'skew', 'curtosis', 'entropy', 'labels']
df_train.columns = column
df_test.columns = column

X_train = np.array(df_train[['var', 'skew', 'curtosis', 'entropy']])
X_test = np.array(df_test[['var', 'skew', 'curtosis', 'entropy']])
y_train = np.array([-1 if y == 0 else 1 for y in df_train['labels']])
y_test = np.array([-1 if y == 0 else 1 for y in df_test['labels']])

one_column = np.ones(X_train.shape[0])
X_train_aug = np.column_stack((X_train, one_column))

one_column = np.ones(X_test.shape[0])
X_test_aug = np.column_stack((X_test, one_column))


def predict(X, W):
    y_pred = np.matmul(X, W)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1
    return y_pred


def average_error(y_true, y_pred):
    error = sum(abs(y_true - y_pred) / 2) / len(y_true)
    return error


def calculate_w_sch_a(X, y, lr, C, no_epoch):
    N, no_features = X.shape
    W = np.zeros(no_features)
    index = np.arange(N)

    for t in range(no_epoch):
        np.random.shuffle(index)
        X = X[index, :]
        y = y[index]

        for i in range(N):
            g = np.copy(W)
            g[no_features - 1] = 0
            if y[i] * np.dot(W, X[i]) <= 1:
                g = g - C * N * y[i] * X[i, :]
            lr = lr / (1 + lr * t / a)
            W = W - lr * g
    return W


def calculate_w_sch(X, y, lr, C, no_epoch):
    N, no_features = X.shape
    W = np.zeros(no_features)
    index = np.arange(N)

    for t in range(no_epoch):
        np.random.shuffle(index)
        X = X[index, :]
        y = y[index]

        for i in range(N):
            g = np.copy(W)
            g[no_features - 1] = 0
            if y[i] * np.dot(W, X[i]) <= 1:
                g = g - C * N * y[i] * X[i, :]
            lr = lr / (1 + t)
            W = W - lr * g
    return W


def constraint(alpha, y):
    return np.matmul(alpha, y)


def obj_fun(alpha, X, y):
    alphayx = np.multiply(alpha[:, None] * y[:, None], X)
    total_loss = 0.5 * np.sum(np.matmul(alphayx, alphayx.T)) - np.sum(alpha)
    return total_loss


def dual_svm(X, y, C):
    N = X.shape[0]
    bounds = [(0, C)] * N
    constraints = {'type': 'eq', 'fun': lambda alpha: constraint(alpha, y)}
    alpha0 = np.zeros(N)

    optimum = optimize.minimize(
        lambda alpha: obj_fun(alpha, X, y), alpha0, method='SLSQP', bounds=bounds, constraints=constraints
    )
    W = np.sum(optimum.x[:, None] * y[:, None] * X, axis=0)
    index = np.where((optimum.x > 0) & (optimum.x < C))
    b = np.mean(y[index] - np.matmul(X[index], W))
    W = np.append(W, b)
    return W


def gaussian_kernel(X1, X2, gamma):
    N1, N2 = X1.shape[0], X2.shape[0]
    Xi = np.tile(X1, (N2, 1, 1)).reshape(-1, X1.shape[1])
    Xj = np.repeat(X2, N1, axis=0)
    return np.exp(-np.sum((Xi - Xj) ** 2, axis=1) / gamma).reshape(N1, N2)


def obj_fun_gaussian(alpha, K, y):
    total_loss = 0.5 * np.sum((alpha[:, None] * y[:, None]) @ K @ (alpha[:, None] * y[:, None]).T) - np.sum(alpha)
    return total_loss


def dual_svm_gaussian(X, y, C, gamma):
    N = X.shape[0]
    bounds = [(0, C)] * N
    constraints = {'type': 'eq', 'fun': lambda alpha: constraint(alpha, y)}
    alpha0 = np.zeros(N)
    K = gaussian_kernel(X, X, gamma)

    optimum = optimize.minimize(
        lambda alpha: obj_fun_gaussian(alpha, K, y), alpha0, method='SLSQP', bounds=bounds, constraints=constraints
    )
    return optimum.x


def predict_dual_svm_gaussian(alpha, X0, y0, X, gamma):
    K = gaussian_kernel(X0, X, gamma)
    pred = np.sign(np.sum(alpha[:, None] * y0[:, None] * K, axis=0))
    return pred


def perceptron_gaussian(X_train, y_train, gamma, no_epoch):
    c = np.zeros(X_train.shape[0])
    K = gaussian_kernel(X_train, X_train, gamma)

    for _ in range(no_epoch):
        for i in range(X_train.shape[0]):
            if np.sign(np.sum(c * y_train * K[:, i])) != y_train[i]:
                c[i] += 1
    return c


def predict_perceptron_gaussian(X_test, c, X_train, y_train, gamma):
    y_pred = np.sign(
        np.array([
            np.sum(c * y_train * np.exp(-np.linalg.norm(x - X_train, axis=1) ** 2 / gamma))
            for x in X_test
        ])
    )
    return y_pred


def kernel_perceptron(X_train, y_train, gamma, no_epoch):
    """Kernel Perceptron"""
    c = np.zeros(X_train.shape[0])
    K = gaussian_kernel(X_train, X_train, gamma)
    for _ in range(no_epoch):
        for i in range(X_train.shape[0]):
            prediction = np.sign(np.sum(c * y_train * K[:, i]))
            if prediction != y_train[i]:
                c[i] += 1
    return c, X_train, y_train


def predict_kernel_perceptron(X_test, c, X_train, y_train, gamma):
    K = gaussian_kernel(X_train, X_test, gamma)
    return np.sign(np.sum(c[:, None] * y_train[:, None] * K, axis=0))


