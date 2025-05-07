import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset():
    """
    Loads the WDBC dataset and preprocesses it.
    See "Loading Dataset" in README"
    """
    dataset_wdbc = pd.read_csv("wdbc.data")
    dummy = pd.DataFrame(dataset_wdbc.columns).T
    dataset_wdbc.columns = range(0, dataset_wdbc.shape[1])
    dataset_wdbc = pd.concat([dataset_wdbc, dummy], axis=0)
    dataset_wdbc.reset_index(inplace=True, drop=True)

    cols = [ 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
    cols = cols*3
    col_names = [f'{i}_mean' for i in cols[:10]]
    col_names += [f'{i}_se' for i in cols[10:20]]
    col_names += [f'{i}_worst' for i in cols[20:]]
    col_names = ['ID', 'Target'] + col_names

    dataset_wdbc.columns = col_names
    for col in dataset_wdbc.columns[2:]:
        dataset_wdbc[col] = dataset_wdbc[col].astype(float)

    X, t = dataset_wdbc.iloc[:, 2:], dataset_wdbc.iloc[:, 1]

    # print(dataset_wdbc.info())

    X = X.to_numpy()
    t = t.to_numpy()

    # Normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    t = le.fit_transform(t)

    # Convert to binary
    t = np.where(t == 1, 1, 0)
    # X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)
    # print("Size of training set:", X_train.shape)
    # print("Size of test set:", X_test.shape)
    return X, t
# , X_train, X_test, t_train, t_test

def plot_hyperparameter_heatmap(results):
    results_df = pd.DataFrame(results, columns=['Learning Rate', 'Epochs', 'Accuracy'])
    heatmap_data = results_df.pivot(index='Epochs', columns='Learning Rate', values='Accuracy')
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Accuracy'})
    plt.title('Hyperparameter Tuning Results')
    plt.xlabel('Learning Rate')
    plt.ylabel('Epochs')
    plt.savefig('logistic_regression_hyperparameter_tuning.png')
    plt.clf()