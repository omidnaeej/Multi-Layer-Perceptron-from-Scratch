import numpy as np
import logging
from models.model import FeedforwardNeuralNetwork
from utils.metrics import accuracy
from utils.visualization import plot_confusion_matrix, plot_error_samples, plot_predictions_vs_actual

def evaluate_model(X_val, Y_val, X_test, Y_test, save_path="models/saved_models/mnist_ffnn.npz", config=None):
    params = np.load(save_path)

    if config is None:
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "layer_sizes": [784, 32, 10],  # Input layer (784), Hidden layer (32), Output layer (10)
            "activations": ["relu", "softmax"],
            "loss": "CrossEntropy",
            "epochs": 20
        }

    # Initialize model and load weights
    model = FeedforwardNeuralNetwork(
        layer_sizes=config["layer_sizes"],
        activations=config["activations"],
        loss=config["loss"]
    )
    model.parameters = {key: params[key] for key in params.files}

    # Predict and evaluate
    Y_pred = model.predict(X_val)
    Y_true = np.argmax(Y_val, axis=0)
    val_accuracy = accuracy(Y_true, Y_pred)

    logging.info(f"Val Accuracy: {100*val_accuracy:.2f}%")
    print(f"Val Accuracy: {100*val_accuracy:.2f}%")
    
    plot_confusion_matrix(y_true = np.argmax(Y_test, axis=0), 
                          y_pred = model.predict(X_test), 
                          class_names = np.unique(Y_test).astype(int))
    
    plot_error_samples(X_val, Y_true, Y_pred, class_names = np.unique(Y_val).astype(int))
    
def evaluate_regression_model(X_test, Y_test, save_path="models/saved_models/life_expectancy_ffnn.npz", config=None):
    params = np.load(save_path)

    if config is None:
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "layer_sizes": [21, 16, 1],
            "activations": ["relu", "linear"],
            "loss": "MSE",
            "epochs": 1000,
            "l1_lambda": 0,
            "l2_lambda": 0
        }

    # Initialize model and load weights
    model = FeedforwardNeuralNetwork(
        layer_sizes=config["layer_sizes"],
        activations=config["activations"],
        loss=config["loss"]
    )
    model.parameters = {key: params[key] for key in params.files}

    # Forward pass to get predictions
    y_pred, _ = model.forward(X_test)

    # Compute MSE
    mse = model._compute_loss(y_pred, Y_test)
    logging.info(f"Test MSE: {mse:.4f}")
    print(f"Test MSE: {mse:.4f}")
    
    plot_predictions_vs_actual(Y_test, y_pred, target_name="Life Expectancy")
    
    return mse

    
    