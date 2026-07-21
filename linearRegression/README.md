# Linear Regression From Scratch

Implementation of Linear Regression using Gradient Descent without relying on Scikit-Learn or PyTorch optimizers.

---

# Overview

Linear Regression models the relationship between input features and a continuous target by learning a linear function

## Mathematical Formulation

The objective of Linear Regression is to learn the optimal parameters **W** and **b** that minimize the prediction error over the training data.

### Prediction

$$
\hat{\mathbf{y}} = \mathbf{XW} + b
$$

where:

- $\mathbf{X}$ : Input feature matrix
- $\mathbf{W}$ : Weight vector
- $b$ : Bias term
- $\hat{\mathbf{y}}$ : Predicted output vector

---

### Loss Function (Mean Squared Error)

$$
L(\mathbf{W}, b) =
\frac{1}{N}
\sum_{i=1}^{N}
\frac{\left(y_i-\hat{y}_i\right)^2}{2}
$$

where:

- $N$ : Number of training samples
- $y_i$ : Ground truth value for sample $i$
- $\hat{y}_i$ : Predicted value for sample $i$

---

### Gradient Descent Weight Update

$$
\mathbf{W}
\leftarrow
\mathbf{W}
-
\eta
\frac{\partial L}{\partial \mathbf{W}}
$$

---

### Bias Update

$$
b
\leftarrow
b
-
\eta
\frac{\partial L}{\partial b}
$$

where:

- $\eta$ : Learning rate

---

The model is trained by minimizing the Mean Squared Error (MSE) loss using a custom implementation of **Mini-Batch Stochastic Gradient Descent (SGD)**. During each training iteration, gradients are computed via PyTorch's automatic differentiation, while parameter updates are performed manually without using `torch.optim`.

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
