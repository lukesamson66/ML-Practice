from utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


def train_softmax_regression(X_train, t_train, X_val, t_val, hyperparams):
    """
    Inspired by my own implementation in coding assignment 2
    """
    N_train = X_train.shape[0]
    MaxEpoch = hyperparams[0]
    learning_rate = hyperparams[1]
    batch_size = hyperparams[2]
    decay = hyperparams[3]

    C = np.max(t_train) + 1

    assert np.min(t_train) == 0 and np.max(t_train) < C
    assert np.min(t_val) == 0 and np.max(t_val) < C
    assert np.min(t_test) == 0 and np.max(t_test) < C

    W = np.random.randn(X_train.shape[1], C) * 0.01
    train_losses = []
    valid_accs = []
    epoch_best = 0
    acc_best = 0
    w_best = None
    stopping_condition = 1
    no_improvement = 0

    for epoch in range(MaxEpoch):
        index = np.random.permutation(N_train)
        X_train_shuffled = X_train[index]
        t_train_shuffled = t_train[index]

        for i in range(0, N_train, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            t_batch = t_train_shuffled[i:i + batch_size]
            t_batch = t_batch.flatten().astype(int)
            
            y_batch, _, loss, _ = predict_softmax(X_batch, W)
            T = np.zeros((X_batch.shape[0], C))
            T[np.arange(X_batch.shape[0]), t_batch] = 1

            grad_W = np.dot(X_batch.T,(y_batch - T))
            grad_W /= len(t_batch)
            grad_W += decay * W 

            W -= learning_rate * grad_W

        _, _, train_loss, _ = predict_softmax(X_train, W, t_train)
        _, _, _, valid_acc = predict_softmax(X_val, W, t_val)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)

        if valid_acc > acc_best:
            acc_best = valid_acc
            epoch_best = epoch
            w_best = W.copy()
        else:
            no_improvement += 1
            if no_improvement >= stopping_condition:
                break
        
  
    return epoch_best, acc_best, w_best, train_losses, valid_accs


def predict_softmax(X, W, t=None):
    """
    Inspired by my own implementation in coding assignment 2
    """
    Z = np.dot(X, W)
    Z -= np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    y = expZ / np.sum(expZ, axis=1, keepdims=True)

    t_hat = np.argmax(y, axis=1)

    if t is None:
        return y, t_hat, None, None
    
    N = X.shape[0]
    T = np.zeros_like(y)
    T[np.arange(N), t] = 1

    loss = -np.sum(T * np.log(y + 1e-16)) / N
    acc = np.mean(t_hat == t)

    return y, t_hat, loss, acc


if __name__ == "__main__":

    X, t = load_dataset()

    X_train, X_temp, t_train, t_temp = train_test_split(X, t, test_size=0.2)
    X_val, X_test, t_val, t_test = train_test_split(X_temp, t_temp, test_size=0.5)

    pca = PCA(n_components=10)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    t_val = t_val.flatten().astype(int)
    t_train = t_train.flatten().astype(int)
    t_test = t_test.flatten().astype(int)
    epoch_list = [10, 50, 100]
    learning_rate_list = [0.001, 0.01, 0.1, 1]
    batch_size_list = [32, 64, 128]
    decay_list = [0.01, 0.1, 0.5, 1.0]
    best_accuracy = 0
    best_params = None
    best_w = None
    results = []
    best_train_losses = None
    best_valid_accs = None
    j = 0
    for epoch in epoch_list:
        for learning_rate in learning_rate_list:
            for batch_size in batch_size_list:
                for decay in decay_list:
                    epoch_best, acc_best, w_best, train_losses, valid_accs = train_softmax_regression(X_train, t_train, X_val, t_val, (epoch, learning_rate, batch_size, decay))
                    results.append((learning_rate, epoch, acc_best))
                    if acc_best > best_accuracy:
                        best_accuracy = acc_best
                        best_params = (epoch, learning_rate, batch_size, decay)
                        best_w = w_best
                        best_train_losses = train_losses
                        best_valid_accs = valid_accs
                    j += 1
    print(f"Best parameters: Epoch: {best_params[0]}, Learning rate: {best_params[1]}, Batch size: {best_params[2]}, Decay: {best_params[3]}")
    print(f"Best validation accuracy: {best_accuracy}")

    plot_train_losses = np.array(best_train_losses)
    plt.plot(plot_train_losses)
    plt.title('Training Losses with Best Hyperparameters')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('softmax_train_losses.png')
    plt.clf()

    plot_valid_accs = np.array(best_valid_accs)
    plt.plot(plot_valid_accs)
    plt.title('Validation Accuracies with Best Hyperparameters')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('softmax_valid_accs.png')
    plt.clf()

    MaxEpoch = best_params[0]
    learning_rate = best_params[1]
    batch_size = best_params[2]
    decay = best_params[3]

    _, _, _, acc_test = predict_softmax(X_test, w_best, t_test)
    print(f"Test accuracy: {acc_test}")

