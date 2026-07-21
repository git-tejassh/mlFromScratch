# Linear Regression From Scratch

Implementation of Linear Regression using Gradient Descent without relying on Scikit-Learn or PyTorch optimizers.

---

# Overview

Linear Regression models the relationship between input features and a continuous target by learning a linear function

\[
\hat{y}=XW+b
\]

where

- **X** : input features
- **W** : learnable weights
- **b** : bias
- **ŷ** : predicted output

The model is trained by minimizing Mean Squared Error using manually implemented Stochastic Gradient Descent.

---

# Features

- Manual weight initialization
- Manual forward propagation
- Mean Squared Error loss
- Manual backpropagation using PyTorch autograd
- Custom SGD optimizer
- Custom DataLoader
- Training & validation loop
- Prediction API
- Loss visualization

---

# Folder Structure

```text
linearRegression/

├── README.md
├── linearRegression_fromScratch.py
├── linearRegression_scratch.ipynb
└── dataLoader.py
```

---

# Components

## LinearRegressionFromScratch

Responsible for

- model initialization
- forward pass
- loss computation
- training
- evaluation
- prediction
- plotting

---

## SGD

Custom optimizer implementing

\[
W = W - \eta \nabla_W
\]

without using `torch.optim`.

---

## dataLoader

Simple mini-batch loader that

- splits dataset
- creates train/test batches
- converts batches into tensors

---

# Training Pipeline

```text
Dataset
      │
      ▼
Data Loader
      │
      ▼
Mini Batches
      │
      ▼
Forward Pass
      │
      ▼
Loss Computation
      │
      ▼
Gradient Calculation
      │
      ▼
Custom SGD Update
      │
      ▼
Repeat
```

---

# Mathematical Formulation

Prediction

\[
\hat y = XW+b
\]

Loss

\[
L=\frac1N\sum \frac{(y-\hat y)^2}{2}
\]

Gradient Descent

\[
W=W-\eta \frac{\partial L}{\partial W}
\]

Bias Update

\[
b=b-\eta \frac{\partial L}{\partial b}
\]

---

# Hyperparameters

| Parameter     | Description                   |
| ------------- | ----------------------------- |
| Learning Rate | Gradient Descent step size    |
| Epochs        | Number of training iterations |
| Batch Size    | Samples per update            |
| Sigma         | Weight initialization std     |

---

# Current Implementation

| Feature               | Status |
| --------------------- | ------ |
| Weight Initialization | ✅     |
| Forward Pass          | ✅     |
| MSE Loss              | ✅     |
| Gradient Descent      | ✅     |
| SGD Optimizer         | ✅     |
| Mini-Batch Training   | ✅     |
| Validation Loop       | ✅     |
| Prediction            | ✅     |
| Loss Plot             | ✅     |

---

# Possible Improvements

- Momentum
- Adam Optimizer
- Early Stopping
- Learning Rate Scheduler
- Model Saving
- Multiple Initialization Methods
- L1 Regularization
- L2 Regularization
- R² Score
- MAE/RMSE Metrics
- Gradient Checking

---

# Complexity

| Operation        | Complexity |
| ---------------- | ---------- |
| Forward Pass     | O(nd)      |
| Backward Pass    | O(nd)      |
| Parameter Update | O(d)       |

where

- n = batch size
- d = number of features

---

# Learning Outcomes

This implementation demonstrates:

- Matrix-based linear algebra
- Gradient Descent optimization
- Parameter initialization
- Loss minimization
- Mini-batch training
- Basic optimizer design

---

# Next Algorithm

➡ Logistic Regression
