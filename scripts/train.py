import numpy as np
import logging
from models.model import FeedforwardNeuralNetwork

def train_model(X_train, Y_train, X_val, Y_val, 
                save_path="models/saved_models/mnist_ffnn.npz", 
                config=None, load_file=None):
    
    if config is None:
        config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "layer_sizes": [784, 32, 10],
            "activations": ["relu", "softmax"],
            "loss": "CrossEntropy",
            "epochs": 20,
            "l1_lambda": 0,
            "l2_lambda": 0
        }

    # Set up logging
    logging.basicConfig(filename="config/train.log", level=logging.INFO, format="%(asctime)s - %(message)s")

    # Initialize model
    model = FeedforwardNeuralNetwork(
        layer_sizes=config["layer_sizes"],
        activations=config["activations"],
        loss=config["loss"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        load_weights_file=load_file
    )

    print(f"Starting training model: {config}")

    # Train model
    logging.info("Starting training...")
    model.train(X_train, Y_train, X_val, Y_val, epochs=config["epochs"])
    logging.info(f"Training completed. Configuration: {config}")

    # Save trained parameters
    # np.savez(save_path, **model.parameters)
    model.save_weights(save_path)
    logging.info(f"Training completed. Model saved successfully at {save_path}")

def train_life_expectancy_predictor(X_train, y_train, X_test, y_test, 
                                    save_path="models/saved_models/life_expectancy_ffnn.npz", 
                                    config=None):
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

    # Set up logging
    logging.basicConfig(filename="config/train.log", level=logging.INFO, format="%(asctime)s - %(message)s")

    # Initialize model
    model = FeedforwardNeuralNetwork(
        layer_sizes=config["layer_sizes"],
        activations=config["activations"],
        loss=config["loss"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
    )

    print(f"Starting training model: {config}")

    # Train model
    logging.info("Starting training...")
    model.train(X_train, y_train, X_test, y_test, epochs=config["epochs"], classification=False)
    logging.info(f"Training completed. Configuration: {config}")

    # Save trained parameters
    np.savez(save_path, **model.parameters)
    logging.info(f"Training completed. Model saved successfully at {save_path}")
