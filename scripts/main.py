import numpy as np
import yaml 
from scripts.train import train_model, train_life_expectancy_predictor
from scripts.evaluate import evaluate_model, evaluate_regression_model
from data.data_loader import load_mnist, get_svhn, load_life_expectancy_dataset
from utils.visualization import plot_sample_images, hist_class_distribution

def load_config(config_file="config/config.yaml"):
    """
    Load configuration from a YAML file.
    
    :param config_file: Path to the YAML configuration file.
    :return: Dictionary containing the configuration.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

def mnist_classifier():
    print(f"Loading MNIST dataset...")
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()

    plot_sample_images(X_train, Y_train)
    hist_class_distribution(np.concatenate((Y_train,Y_val,Y_test), axis=1))

    config = load_config()

    train_model(X_train, Y_train, X_val, Y_val, 
                save_path="models/saved_models/mnist_ffnn.npz", 
                config=config)

    evaluate_model(X_val, Y_val, X_test, Y_test, 
                   save_path="models/saved_models/mnist_ffnn.npz", 
                   config=config)

def svhn_classifier():
    print(f"Loading SVHN dataset...")
    X_train, Y_train, X_test, Y_test = get_svhn()
    
    plot_sample_images(X_train, Y_train)
    
    config = load_config()

    train_model(X_train, Y_train, X_test, Y_test, 
                save_path="models/saved_models/svhn_ffnn.npz", 
                config=config)

    evaluate_model(X_test, Y_test, X_test, Y_test, 
                   save_path="models/saved_models/svhn_ffnn.npz", 
                   config=config)

def svhn_classifier_transfer_learning():
    print(f"Loading SVHN dataset...")
    X_train, Y_train, X_test, Y_test = get_svhn()
    
    plot_sample_images(X_train, Y_train)
    
    config = load_config()

    train_model(X_train, Y_train, X_test, Y_test, 
                save_path="models/saved_models/svhn_ffnn.npz", 
                config=config, 
                load_file="models/saved_models/mnist_ffnn.npz")

    evaluate_model(X_test, Y_test, X_test, Y_test, 
                   save_path="models/saved_models/svhn_ffnn.npz", 
                   config=config)

def life_expectancy_predictor():
    print(f"Loading Life Expectancy dataset...")
    X_train, Y_train, X_test, Y_test = load_life_expectancy_dataset()
    
    config = load_config("config/q2_config.yaml")
    
    train_life_expectancy_predictor(X_train, Y_train, X_test, Y_test, 
                                    save_path="models/saved_models/life_expectancy_ffnn.npz", 
                                    config=config)

    evaluate_regression_model(X_test, Y_test, 
                              save_path="models/saved_models/life_expectancy_ffnn.npz", 
                              config=config)



# Q1-2
mnist_classifier()

# # Q1-3
svhn_classifier()
svhn_classifier_transfer_learning()

# Q2
life_expectancy_predictor()


