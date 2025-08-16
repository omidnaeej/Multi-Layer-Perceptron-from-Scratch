import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def hist_class_distribution(y):
    y = np.argmax(y, axis=0)  # Convert one-hot encoded labels back to integers
    unique, counts = np.unique(y, return_counts=True)

    plt.bar(unique, counts)
    plt.xlabel("Digit Class")
    plt.ylabel("Number of Samples")
    plt.title("MNIST Digit Class Distribution")
    plt.xticks(range(10), labels=[str(i) for i in range(10)])

    for i in range(len(unique)):
        plt.text(unique[i], counts[i], str(counts[i]), ha='center', va='bottom', fontsize=10)

    plt.show()

def plot_loss_curve(train_losses, val_losses, loss):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel(f"Loss ({loss})")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()
    
def plot_loss_acc(train_losses, val_losses, train_accuracies, val_accuracies):
    # Plot the training loss curve
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.suptitle('Training & Validation Metrices')
    plt.show()
    
def plot_sample_images(X, y):
    y = np.argmax(y, axis=0)  # Convert one-hot encoded labels back to integers
    
    plt.figure(figsize=(10, 10))
    
    for class_label in range(10):
        class_indices = np.where(y == class_label)[0]
        
        # Randomly select 10 images from the current class
        random_indices = np.random.choice(class_indices, size=10, replace=False)
        
        for i, idx in enumerate(random_indices):
            plt.subplot(10, 10, class_label * 10 + i + 1)
            plt.imshow(X[:, idx].reshape(28, 28), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title(f"Class {class_label}")
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plots a confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = np.unique(y_true)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names[:], yticklabels=class_names[:])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
def plot_error_samples(X, Y_true, Y_pred, class_names=None):
    """
    Plot 16 misclassified images with their true and predicted labels.
    """
    # Convert one-hot encoded labels to class indices if necessary
    if Y_true.ndim > 1:  
        true_labels = np.argmax(Y_true, axis=0)
    else: 
        true_labels = Y_true
    
    if Y_pred.ndim > 1: 
        pred_labels = np.argmax(Y_pred, axis=0)
    else: 
        pred_labels = Y_pred
    
    misclassified = np.where(true_labels != pred_labels)[0]
    
    if len(misclassified) == 0:
        print("No misclassified samples found.")
        return
    
    # Randomly select 16 misclassified samples
    num_samples = min(16, len(misclassified))
    selected_indices = np.random.choice(misclassified, num_samples, replace=False)
    
    if class_names is None:
        class_names = [str(i) for i in range(np.max(true_labels) + 1)]
        
    # print(true_labels, pred_labels, true_labels.shape, pred_labels.shape)
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(selected_indices):
        plt.subplot(4, 4, i + 1)
        plt.imshow(X[:, idx].reshape(28, 28), cmap="gray")
        plt.title(f"True: {true_labels[idx]}\nPred: {pred_labels[idx]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
 
def plot_predictions_vs_actual(y_test, y_pred, target_name="Life Expectancy"):
    plt.figure(figsize=(8, 6))
    plt.plot([min(y_test), min(y_pred)], [max(y_test), max(y_pred)], 'r--', label='Predicted = Actual', linewidth=1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel(f"Actual {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    plt.title(f"Actual vs Predicted {target_name}")
    plt.show()
    
    