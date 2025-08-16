import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import scipy.io as sio
from PIL import Image
import requests
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_mnist():
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    y = y.astype(np.int32)
    
    print("Image shape:", X.shape) # Check the shape of a single image

    # Standardization
    # mean = np.mean(X, axis=0)
    # std = np.std(X, axis=0)+ 1e-10
    # X = (X - mean) / std

    # Normalize the pixel values to be in the range [0, 1]
    X = X.astype(np.float32) / 255.0
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, shuffle=True)
    
    # One-hot encode labels
    y_train = np.eye(10)[y_train].T.astype(np.float32)  # Convert labels to one-hot (10, samples)
    y_test = np.eye(10)[y_test].T.astype(np.float32)
    y_val = np.eye(10)[y_val].T.astype(np.float32)
    
    # Ensure X_train, X_val, and X_test are NumPy arrays
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
        
    # Ensure Y_train is not Pandas DataFrame
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.to_numpy()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy()
    if isinstance(y_val, pd.DataFrame):
        y_val = y_val.to_numpy()

    # print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    # print(f"Y_train shape: {y_train.shape}, Y_val shape: {y_val.shape}")
                
    return X_train.T, y_train, X_val.T, y_val, X_test.T, y_test  

def download_svhn(data_dir="data/svhn"):
    """
    Download the SVHN dataset if it doesn't already exist.
    """
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
    test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    
    # Download train data
    train_path = os.path.join(data_dir, "train_32x32.mat")
    if not os.path.exists(train_path):
        print("Downloading training data...")
        response = requests.get(train_url, stream=True)
        with open(train_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    
    # Download test data
    test_path = os.path.join(data_dir, "test_32x32.mat")
    if not os.path.exists(test_path):
        print("Downloading test data...")
        response = requests.get(test_url, stream=True)
        with open(test_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def load_svhn(data_dir="data/svhn"):
    """
    Load the SVHN dataset from .mat files.
    """

    # Load training data
    train_data = sio.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    X_train = train_data["X"]  
    Y_train = train_data["y"]
    
    # Load test data
    test_data = sio.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    X_test = test_data["X"]  
    Y_test = test_data["y"]
    
    # Convert labels from (num_samples, 1) to (num_samples,)
    Y_train = Y_train.flatten()
    Y_test = Y_test.flatten()
    
    # Convert labels from 1-based to 0-based indexing
    Y_train[Y_train == 10] = 0
    Y_test[Y_test == 10] = 0
    
    # Transpose images to (num_samples, height, width, channels)
    X_train = np.transpose(X_train, (3, 0, 1, 2))
    X_test = np.transpose(X_test, (3, 0, 1, 2))
    
    return X_train, Y_train, X_test, Y_test

def preprocess_svhn(X_train, Y_train, X_test, Y_test, resize_shape=(28, 28)):
    """
    Preprocess the SVHN dataset.
    """
    # Resize images to 28x28, convert to grayscale, and flatten them
    def resize_to_grayscale_and_flatten(images, resize_shape):
        resized_images = []
        for img in images:
            img_pil = Image.fromarray((img * 255).astype("uint8"))

            img_resized = img_pil.resize(resize_shape, Image.BILINEAR)
            img_grayscale = img_resized.convert("L")
            img_flattened = np.array(img_grayscale).flatten()

            resized_images.append(img_flattened)
        return np.array(resized_images)
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    # Resize, convert to grayscale, and flatten images
    X_train = resize_to_grayscale_and_flatten(X_train, resize_shape)
    X_test = resize_to_grayscale_and_flatten(X_test, resize_shape)
    
    Y_train = np.eye(10)[Y_train].T.astype(np.float32)  # Convert labels to one-hot (10, samples)
    Y_test = np.eye(10)[Y_test].T.astype(np.float32)
    
    # Split training data into training and validation sets
    # X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_size, random_state=42)
    
    return X_train, Y_train, X_test, Y_test

def get_svhn(data_dir="data/svhn"):
    """
    Download, load, and preprocess the SVHN dataset.
    """
    download_svhn(data_dir)
    X_train, Y_train, X_test, Y_test = load_svhn(data_dir)
    X_train, Y_train, X_test, Y_test = preprocess_svhn(X_train, Y_train, X_test, Y_test)
    
    # print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    # print(f"Y_train shape: {Y_train.shape}, Y_test shape: {Y_test.shape}")
    return X_train.T, Y_train, X_test.T, Y_test

def load_life_expectancy_dataset(path="data/life expectancy/Life Expectancy Data.csv"):
    data = pd.read_csv(path)

    print("Missing values:", data.isnull().sum())

    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns

    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    for col in categorical_columns:
        data[col] = data[col].fillna(data[col].mode()[0])
        
    print("After filling missing values:", data.isnull().sum())
     

    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    
    print(np.unique(data["Year"]))

    train_data = data[data["Year"] <= 2010]
    test_data = data[data["Year"] > 2010]

    scaler = StandardScaler()
    train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
    test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])
    
    X_train = train_data.drop(columns=["Life expectancy "]).values 
    y_train = train_data["Life expectancy "].values 
    X_test = test_data.drop(columns=["Life expectancy "]).values 
    y_test = test_data["Life expectancy "].values

    y_train = y_train.reshape(1, -1)  # Reshape y_train to (1, num_samples)
    y_test = y_test.reshape(1, -1)
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train.T, y_train, X_test.T, y_test
