# Logistic Regression From Scratch

Implementation of Logistic Regression using Gradient Descent without relying on Scikit-Learn or PyTorch optimizers.

---

# Overview

Logistic Regression models the probability that an input belongs to a given class by passing a linear combination of features through a sigmoid function.

## Mathematical Formulation

The objective of Logistic Regression is to learn the optimal parameters **W** and **b** that minimize the Binary Cross-Entropy loss over the training data.

### Prediction

$$
\hat{\mathbf{y}} = \sigma(\mathbf{XW} + b) = \frac{1}{1 + e^{-(\mathbf{XW} + b)}}
$$

where:

- $\mathbf{X}$ : Input feature matrix
- $\mathbf{W}$ : Weight vector
- $b$ : Bias term
- $\sigma$ : Sigmoid activation function
- $\hat{\mathbf{y}}$ : Predicted probability vector

---

### Loss Function (Binary Cross-Entropy)

$$
L(\mathbf{W}, b) =
-\frac{1}{N}
\sum_{i=1}^{N}
\left[
y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
\right]
$$

where:

- $N$ : Number of training samples
- $y_i$ : Ground truth label for sample $i$
- $\hat{y}_i$ : Predicted probability for sample $i$

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

The model is trained by minimizing the Binary Cross-Entropy loss using a custom implementation of **Mini-Batch Stochastic Gradient Descent (SGD)**. During each training iteration, gradients are computed via PyTorch's automatic differentiation, while parameter updates are performed manually without using `torch.optim`.

# Features

- Manual weight initialization
- Manual forward propagation with sigmoid activation
- Binary Cross-Entropy loss
- Manual backpropagation using PyTorch autograd
- Custom SGD optimizer
- Custom DataLoader
- Training & validation loop
- Prediction API
- Loss visualization

---

# Folder Structure

```text
logisticRegression/

├── README.md
├── logisticRegression_fromScratch.py
└── logisticRegression_scratch.ipynb
```

---

# Components

## LogisticRegressionFromScratch

Responsible for

- model initialization
- forward pass (linear transform + sigmoid)
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
Forward Pass (Linear + Sigmoid)
      │
      ▼
Binary Cross-Entropy Loss
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
\hat y = \sigma(XW+b)
\]

Loss

\[
L=-\frac1N\sum \left[y\log(\hat y) + (1-y)\log(1-\hat y)\right]
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

| Feature                 | Status |
| ----------------------- | ------ |
| Weight Initialization   | ✅     |
| Forward Pass (Sigmoid)  | ✅     |
| Binary Cross-Entropy    | ✅     |
| Gradient Descent        | ✅     |
| SGD Optimizer           | ✅     |
| Mini-Batch Training     | ✅     |
| Validation Loop         | ✅     |
| Prediction              | ✅     |
| Loss Plot               | ✅     |
| Classification Accuracy | ⏳     |
| Decision Boundary Plot  | ⏳     |

---

# Possible Improvements

- Classification accuracy / precision / recall / F1 metrics
- Decision boundary visualization
- Confusion matrix
- Momentum
- Adam Optimizer
- Early Stopping
- Learning Rate Scheduler
- Model Saving
- L1 Regularization
- L2 Regularization
- Gradient Checking
- Class imbalance handling

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

- Sigmoid activation and probabilistic interpretation of linear outputs
- Binary Cross-Entropy as a loss function
- Gradient Descent optimization
- Parameter initialization
- Mini-batch training
- Basic optimizer design

---

# Next Algorithm

➡ Softmax Regression
