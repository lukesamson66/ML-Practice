from utils import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from keras import optimizers
from keras.callbacks import EarlyStopping


if __name__ == "__main__":
    X, t = load_dataset()
    X_train, X_temp, t_train, t_temp = train_test_split(X, t, test_size=0.2, random_state=42)
    X_val, X_test, t_val, t_test = train_test_split(X_temp, t_temp, test_size=0.5, random_state=42)

    learning_rates = [0.01, 0.1, 1.0]
    epoch_list = [10, 50, 100]
    hidden_layer_acts = ['relu', 'tanh']
    hidden_units_list = [10, 50, 100]
    output_layer_act = 'sigmoid'
    best_accuracy = 0
    best_model = None
    best_epoch = 0
    best_hidden_units = 0
    best_hidden_layer_act = ''
    best_history = None
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, restore_best_weights=True)
    for hidden_units in hidden_units_list:
        for hidden_layer_act in hidden_layer_acts:
            model = models.Sequential()
            model.add(layers.Dense(hidden_units, input_dim=X_train.shape[1], activation=hidden_layer_act))
            model.add(layers.Dense(hidden_units, activation=hidden_layer_act))
            model.add(layers.Dense(1, activation=output_layer_act))
            for learning_rate in learning_rates:
                sgd = optimizers.SGD(learning_rate=learning_rate)
                model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
                for epochs in epoch_list:
                    history = model.fit(X_train, t_train, epochs=epochs, batch_size=len(X_train), validation_data=(X_val, t_val), verbose=2, callbacks=[early_stopping])
                    val_accuracy = history.history['val_acc'][-1]

                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_model = model
                        best_epoch = epochs
                        best_hidden_units = hidden_units
                        best_hidden_layer_act = hidden_layer_act
                        best_history = history

    predictions = best_model.predict(X_test)

    rounded = [int(round(x[0])) for x in predictions]
    accuracy = np.mean(np.array(rounded) == t_test)
    print("Accuracy:", accuracy)
    print("Best model parameters:")
    print("Learning rate:", learning_rate)
    print("Epochs:", best_epoch)
    print("Hidden units:", best_hidden_units)
    print("Hidden layer activation function:", best_hidden_layer_act)
    
    plt.figure()
    plt.plot(best_history.history['acc'])
    plt.plot(best_history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('neural_accuracy.png')

    plt.figure()
    plt.plot(best_history.history['loss'])
    plt.plot(best_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('neural_loss.png')
