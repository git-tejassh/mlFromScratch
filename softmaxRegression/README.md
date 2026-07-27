# Softmax Regression From Scratch

Implementation of Softmax (Multinomial Logistic) Regression using Gradient Descent, without relying on Scikit-Learn or PyTorch's built-in optimizers/loss modules.

> **Status: Not yet implemented.** This README documents the intended design and math. See the "Current Implementation" table below for what actually exists in the codebase today.

---

# Overview

Softmax Regression generalizes Logistic Regression to multi-class classification. Instead of predicting a single probability via the sigmoid function, it predicts a probability distribution over **K** classes using the softmax function.

The target use case in this repo is classifying **FashionMNIST** images (10 classes).

## Mathematical Formulation

### Prediction

$$
\mathbf{z} = \mathbf{XW} + \mathbf{b}
$$

$$
\hat{y}_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

where:

- $\mathbf{X}$ : Input feature matrix (flattened image pixels)
- $\mathbf{W}$ : Weight matrix (features × classes)
- $\mathbf{b}$ : Bias vector (one per class)
- $K$ : Number of classes (10 for FashionMNIST)
- $\hat{y}_k$ : Predicted probability of class $k$

---

### Loss Function (Categorical Cross-Entropy)

$$
L(\mathbf{W}, \mathbf{b}) =
-\frac{1}{N}
\sum_{i=1}^{N}
\sum_{k=1}^{K}
y_{i,k} \log(\hat{y}_{i,k})
$$

where:

- $N$ : Number of training samples
- $y_{i,k}$ : One-hot encoded ground truth (1 if sample $i$ belongs to class $k$, else 0)
- $\hat{y}_{i,k}$ : Predicted probability of sample $i$ belonging to class $k$

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
\mathbf{b}
\leftarrow
\mathbf{b}
-
\eta
\frac{\partial L}{\partial \mathbf{b}}
$$

where:

- $\eta$ : Learning rate

---

The model will be trained by minimizing Categorical Cross-Entropy loss using a custom implementation of **Mini-Batch Stochastic Gradient Descent (SGD)** (the same `utils/gradientDescent.py` optimizer used by Linear and Logistic Regression). Gradients will be computed via PyTorch autograd; parameter updates will be manual, without `torch.optim`.

---

# Features (Planned)

- Manual weight initialization (weight matrix, not vector — multi-class)
- Manual forward propagation with softmax activation
- Categorical Cross-Entropy loss
- Manual backpropagation using PyTorch autograd
- Custom SGD optimizer (reuse existing `SGD` class)
- FashionMNIST data pipeline
- Training & validation loop
- Prediction API (argmax over class probabilities)
- Accuracy metric
- Loss visualization

---

# Folder Structure

```text
softmaxRegression/

├── README.md
├── main.py                 (currently empty — model not implemented)
└── main_notebook.ipynb      (currently: data loading + visualization only)
```

---

# Components

## SoftmaxRegressionFromScratch (not yet implemented)

Will be responsible for:

- model initialization (weight matrix + bias vector)
- forward pass (linear transform + softmax)
- loss computation (categorical cross-entropy)
- training
- evaluation
- prediction (argmax)
- accuracy calculation
- plotting

## SGD

Reuses the existing optimizer from `utils/gradientDescent.py`:

\[
W = W - \eta \nabla_W
\]

without using `torch.optim`.

## FashionMNIST data pipeline

Currently implemented in the notebook via `torchvision.datasets.FashionMNIST` + `torch.utils.data.DataLoader`, **not** the repo's custom `utils/dataLoader.py`. This is inconsistent with the "from scratch" philosophy used in Linear/Logistic Regression and should be reconciled (either justify using torchvision's loader for image data, or adapt the custom loader to support it).

---

# Training Pipeline (Planned)

```text
FashionMNIST Dataset
      │
      ▼
Data Loader (batching, resize to 32x32, tensor conversion)
      │
      ▼
Mini Batches
      │
      ▼
Forward Pass (Linear + Softmax)
      │
      ▼
Categorical Cross-Entropy Loss
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

# Hyperparameters (Planned)

| Parameter     | Description                   |
| ------------- | ----------------------------- |
| Learning Rate | Gradient Descent step size    |
| Epochs        | Number of training iterations |
| Batch Size    | Samples per update            |
| Sigma         | Weight initialization std     |
| Num Classes   | Number of output classes (10) |

---

# Current Implementation Status

| Feature                                         | Status |
| ----------------------------------------------- | ------ |
| FashionMNIST loading                            | ✅     |
| Data visualization (`show_images`, `visualize`) | ✅     |
| Text label mapping                              | ✅     |
| Model class (weights/bias init)                 | ❌     |
| Forward pass (softmax)                          | ❌     |
| Categorical Cross-Entropy loss                  | ❌     |
| Training loop                                   | ❌     |
| Validation loop                                 | ❌     |
| Custom SGD integration                          | ❌     |
| Prediction API                                  | ❌     |
| Accuracy metric                                 | ❌     |
| Loss plot                                       | ❌     |
| Consistent `data_loader` usage                  | ❌     |

---

# Possible Improvements (Post-Implementation)

- Classification accuracy / precision / recall / F1 metrics
- Confusion matrix
- Momentum
- Adam Optimizer
- Early Stopping
- Learning Rate Scheduler
- Model Saving
- L1/L2 Regularization
- Gradient Checking
- Numerically stable softmax (subtract max logit before exponentiating)

---

# Complexity

| Operation        | Complexity |
| ---------------- | ---------- |
| Forward Pass     | O(ndk)     |
| Backward Pass    | O(ndk)     |
| Parameter Update | O(dk)      |

where

- n = batch size
- d = number of features
- k = number of classes

---

# Learning Outcomes (Target)

This implementation is intended to demonstrate:

- Generalizing binary logistic regression to multi-class classification
- Softmax activation and its numerical stability considerations
- Categorical Cross-Entropy as a loss function
- One-hot encoding of labels
- Gradient Descent optimization for multi-class weight matrices
- Mini-batch training on image data

---

# Next Algorithm

➡ Decision Tree
