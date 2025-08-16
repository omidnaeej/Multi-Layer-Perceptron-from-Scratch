import numpy as np
import pandas as pd
from utils.metrics import accuracy
from utils.visualization import plot_loss_acc, plot_loss_curve

class FeedforwardNeuralNetwork:
    def __init__(self, layer_sizes, activations, loss="MSE", learning_rate=0.001, batch_size=32, l1_lambda=0.0, l2_lambda=0.0, load_weights_file=None):
        """
        Initialize the Feedforward Neural Network.
        
        :param layer_sizes: List containing the sizes of each layer (including input & output).
        :param activations: List of activation functions per layer (excluding input layer).
        :param loss: Loss function to use ('MSE' or 'CrossEntropy').
        :param learning_rate: Learning rate for gradient descent.
        :param batch_size: Batch size for training.
        :param l1_lambda: L1 regularization strength (default: 0.0, no regularization).
        :param l2_lambda: L2 regularization strength (default: 0.0, no regularization).
        :param load_weights_file: Path to a .npz file containing pre-trained weights (optional).
        """
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss = loss
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        if load_weights_file: 
            self.parameters = self.load_weights(load_weights_file)
        else: 
            self.parameters = self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with N(0, 0.1) and biases with 0."""
        np.random.seed(42)
        parameters = {}
        for i in range(1, len(self.layer_sizes)):
            # Initialize weights with mean=0 and variance=0.1
            parameters[f"W{i}"] = np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) * np.sqrt(0.1)
            parameters[f"b{i}"] = np.zeros((self.layer_sizes[i], 1))
        return parameters
    
    def load_weights(self, load_weights_file):
        """
        Load weights from a .npz file.
        """
        try:
            data = np.load(load_weights_file)
            parameters = {}
            for i in range(1, len(self.layer_sizes)):
                parameters[f"W{i}"] = data[f"W{i}"]
                parameters[f"b{i}"] = data[f"b{i}"]
            print("Weights loaded successfully.")
            return parameters
        
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Initializing weights randomly instead.")
            return self._initialize_weights()
        
    def save_weights(self, file_path):
        np.savez(file_path, **self.parameters)
        print(f"Weights saved to {file_path}")

    def _activation_function(self, Z, func, deriv=False):
        """Apply activation functions and their derivatives."""
        if func == "relu":
            return Z * (Z > 0) if not deriv else (Z > 0)
        elif func == "sigmoid":
            A = 1 / (1 + np.exp(-Z))
            return A if not deriv else A * (1 - A)
        elif func == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True)) 
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        elif func == "linear":
            return Z if not deriv else np.ones_like(Z)
        else:
            raise ValueError(f"Unsupported activation function: {func}")

    def forward(self, X):
        """Perform forward propagation."""
        A = X
        caches = {"A0": A}  # Store activations for backprop
        for i in range(1, len(self.layer_sizes)):
            W, b = self.parameters[f"W{i}"], self.parameters[f"b{i}"]
            Z = np.dot(W, A) + b
            A = self._activation_function(Z, self.activations[i-1])
            caches[f"Z{i}"], caches[f"A{i}"] = Z, A
        return A, caches

    def _compute_loss(self, Y_pred, Y_true):
        """
        Compute the loss (MSE or CrossEntropy) with optional L1 and L2 regularization.
        """
        if self.loss == "CrossEntropy":
            Y_pred = np.clip(Y_pred, 1e-8, 1 - 1e-8) 
            loss = -np.mean(np.sum(Y_true * np.log(Y_pred), axis=0))
        elif self.loss == "MSE":
            loss = np.mean(np.sum((Y_true - Y_pred) ** 2, axis=0))  
        else:
            raise ValueError("Unsupported loss function")
        
        # Add L1 regularization
        if self.l1_lambda > 0:
            l1_loss = 0
            for key in self.parameters:
                if key.startswith("W"): 
                    l1_loss += np.sum(np.abs(self.parameters[key]))
            loss += self.l1_lambda * l1_loss
        
        # Add L2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0
            for key in self.parameters:
                if key.startswith("W"): 
                    l2_loss += np.sum(self.parameters[key] ** 2)
            loss += self.l2_lambda * l2_loss
        
        return loss

    def backward(self, X, Y, caches):
        grads = {}
        L = len(self.layer_sizes) - 1 
        m = X.shape[1]  

        # Compute dA for the output layer
        if self.loss == "CrossEntropy":
            if self.activations[-1] == "softmax":
                dZ = caches[f"A{L}"] - Y  # Simplified gradient for softmax + cross-entropy
            else:
                dZ = -(Y / (caches[f"A{L}"] + 1e-8))  # Avoid division by zero
        elif self.loss == "MSE":
            dZ = 2 * (caches[f"A{L}"] - Y) / m
        else:
            raise ValueError("Unsupported loss function")

        # Loop backward through layers
        for i in reversed(range(1, L + 1)):
            if i != L or self.activations[i-1] != "softmax":
                dZ = dZ * self._activation_function(caches[f"Z{i}"], self.activations[i-1], deriv=True)
            
            grads[f"dW{i}"] = np.dot(dZ, caches[f"A{i-1}"].T) / m
            grads[f"db{i}"] = np.sum(dZ, axis=1, keepdims=True) / m
            
            # Add L1 and L2 regularization gradients
            if self.l1_lambda > 0:
                grads[f"dW{i}"] += self.l1_lambda * np.sign(self.parameters[f"W{i}"])
            if self.l2_lambda > 0:
                grads[f"dW{i}"] += self.l2_lambda * 2 * self.parameters[f"W{i}"]

            if i > 1:
                dZ = np.dot(self.parameters[f"W{i}"].T, dZ)  # Update dZ for next layer

        return grads

    def update_parameters(self, grads):
        """Update parameters using gradient descent."""
        for i in range(1, len(self.layer_sizes)):
            self.parameters[f"W{i}"] -= self.learning_rate * grads[f"dW{i}"]
            self.parameters[f"b{i}"] -= self.learning_rate * grads[f"db{i}"]
  
    def train(self, X_train, Y_train, X_val, Y_val, epochs=20, classification=True):
        """
        Train the model without shuffling to reduce time complexity.
        """
        # Arrays to store training and validation metrics
        train_losses = np.empty(epochs)
        train_accuracies = np.empty(epochs)
        val_losses = np.empty(epochs)
        val_accuracies = np.empty(epochs)
        
        m = X_train.shape[1]
        for epoch in range(1, epochs + 1):
            for i in range(0, m, self.batch_size):
                # Extract mini-batch
                X_batch = X_train[:, i:i + self.batch_size]
                Y_batch = Y_train[:, i:i + self.batch_size]

                Y_pred, caches = self.forward(X_batch)
                loss = self._compute_loss(Y_pred, Y_batch)

                grads = self.backward(X_batch, Y_batch, caches)
                self.update_parameters(grads)
                
            # Compute validation loss and accuracy
            Y_val_pred, _ = self.forward(X_val)
            val_loss = self._compute_loss(Y_val_pred, Y_val)
            val_acc = accuracy(np.argmax(Y_val, axis=0), np.argmax(Y_val_pred, axis=0))

            # Compute training loss and accuracy
            Y_true = np.argmax(Y_train, axis=0)
            Y_pred, _ = self.forward(X_train)
            acc = accuracy(Y_true, np.argmax(Y_pred, axis=0))
            
            if classification:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {loss:.4f} - Train Accuracy: {100 * acc:.4f}% - Val Loss: {val_loss:.4f} - Val Accuracy: {100 * val_acc:.4f}%")
            else:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {loss:.4f} - Val Loss: {val_loss:.4f}")
                
            train_losses[epoch - 1] = loss
            train_accuracies[epoch - 1] = acc
            val_losses[epoch - 1] = val_loss
            val_accuracies[epoch - 1] = val_acc
        
        if classification:
            plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies)
        else:
            plot_loss_curve(train_losses, val_losses, self.loss)
                
    def predict(self, X):
        """Make predictions using the trained model."""
        Y_pred, _ = self.forward(X)
        return np.argmax(Y_pred, axis=0)
