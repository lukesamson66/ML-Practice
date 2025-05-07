import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_dataset, plot_hyperparameter_heatmap


def train_logistic_regression(X, t, learning_rate = None, epochs = None):
    """
    Train a logistic regression model using gradient descent.
    Inspired by my own implementation in Coding Assignment 2.
    """
    if learning_rate is None:
        learning_rate = 0.001
    if epochs is None:
        epochs = 10
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        z = np.dot(X,w) + b
        y_pred = 1 / (1 + np.exp(-z))
        error = y_pred - t
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

def predict_logistic_regression(X, w, b):
    """
    Makes a prediction based on logistic regression.
    Inspired by my own implementation in Coding Assignment 2.
    """
    y_pred = np.dot(X,w) + b
    t = (y_pred >= 0.5).astype(int)
    return t

if __name__ == "__main__":
    
    X, t = load_dataset()

    X_train, X_temp, t_train, t_temp = train_test_split(X, t, test_size=0.2)
    X_val, X_test, t_val, t_test = train_test_split(X_temp, t_temp, test_size=0.5)
    
    epoch_list = [10, 50, 100, 200, 500]
    learning_rate_list = [0.001, 0.01, 0.1, 1]
    best_accuracy = 0
    best_lr = None
    best_epochs = None
    best_w = None
    best_b = None
    results = []
    for epochs in epoch_list:
        for lr in learning_rate_list:
            w, b = train_logistic_regression(X_train, t_train, learning_rate=lr, epochs=epochs)
            t_hat_val = predict_logistic_regression(X_val, w, b)
            accuracy = accuracy_score(t_val, t_hat_val)
            results.append((lr, epochs, accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_lr = lr
                best_epochs = epochs
                best_w = w
                best_b = b

    plot_hyperparameter_heatmap(results)

    print(f"Best accuracy: {best_accuracy:.4f} with learning rate: {best_lr} and epochs: {best_epochs}")
    t_hat = predict_logistic_regression(X_test, best_w, best_b)

    accuracy = accuracy_score(t_test, t_hat)

    print(f"Test accuracy of Logistic Regression: {accuracy:.4f}")  

    plt.figure
    plt.barh(range(len(best_w)), best_w, align='center')
    plt.xlabel('Weight')
    plt.ylabel('Feature Index')
    plt.title('Logistic Regression Weights')
    plt.savefig('logistic_regression_weights.png')

