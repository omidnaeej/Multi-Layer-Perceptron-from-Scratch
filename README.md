# Multi-Layer-Perceptron-from-Scratch
This repository contains the implementation of a simple **neural network built entirely from scratch**, as part of Assignment 1 of the Deep Learning course.  
The goal of this project is to better understand the inner workings of neural networks by implementing the key components without relying on deep learning frameworks such as TensorFlow or PyTorch.

---

## Repository Contents
- `config/`: Hyperparameter settings
- `data/`: Data loading module
- `models/`: Neural network implementation
- `notebooks/`: Jupyter notebooks (optional)
- `scripts/`: Training and evaluation scripts
- `utils/`: Metrics and visualization
- `report.pdf` – Original report (in Persian)    

---

## Project Overview
The project demonstrates the construction of a **fully functional neural network pipeline**:
- **Data Loading & Preprocessing**  
  Implemented custom data loaders and dataset handling.  

- **Model Architecture**  
  A feed-forward neural network implemented from scratch with support for activation functions (Sigmoid, ReLU, etc.).  

- **Training Loop**  
  Implementation of forward propagation, backpropagation, and gradient descent optimization.  

- **Evaluation & Metrics**  
  Scripts for evaluating model performance and computing custom metrics.  

- **Visualization**  
  Plotting training curves and evaluation results.  

---

## Setup
Clone the repository:
```bash
git clone https://github.com/<your-username>/Neural-Network-from-Scratch.git
cd Neural-Network-from-Scratch
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script to train and test the neural network:

```bash
python -m scripts.main
```
