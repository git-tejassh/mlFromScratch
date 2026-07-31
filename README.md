# Machine Learning From Scratch

> Building classical machine learning algorithms from first principles using only Python, NumPy, and minimal PyTorch tensor operations.

This repository is a personal learning project where every machine learning algorithm is implemented from scratch without relying on high-level libraries such as Scikit-Learn.

The objective is to understand the mathematics, optimization, and implementation details behind each algorithm instead of treating ML as a black box.

---

# Goals

- Implement every algorithm from scratch
- Understand the underlying mathematics
- Build reusable utilities (optimizers, dataloaders, metrics, etc.)
- Learn how popular ML libraries work internally
- Create a reference repository for interviews and future projects

---

# Repository Structure

```text
MachineLearning-FromScratch/
│
├── README.md
│
│
├── utils/
│   ├── dataloader.py
│   ├── gradientDescent.py
│   ├── utilities.py
│
│
├── linearRegression/
│   ├── README.md
│   ├── linearRegression_fromScratch.py
│   ├── linearRegression_scratch.ipynb
│   
│
├── logisticRegression/
│   ├── README.md
│   ├── logisticRegression.ipynb
│   ├── main.py
│
├── softmaxRegression/
│   ├──README.md
│   ├──main.py
│   ├──main_notebook.ipynb
│
│
├── decisionTree/
│
├── randomForest/
│
├── svm/
│
├── naiveBayes/
│
├── kNearestNeighbors/
│
├── kMeans/
│
├── pca/
│
├── neuralNetworks/
│
├── cnn/
│
├── rnn/
│
├── lstm/
│
└── transformers/
```

---

# Learning Roadmap

## Classical Machine Learning

| Algorithm           | Status      |
| ------------------- | ----------- |
| Linear Regression   | ✅ Complete |
| Logistic Regression | ✅ Complete |
| Softmax Regression  | ✅ Complete |
| Decision Tree       | ⏳ Planned  |
| Random Forest       | ⏳ Planned  |
| Naive Bayes         | ⏳ Planned  |
| KNN                 | ⏳ Planned  |
| SVM                 | ⏳ Planned  |
| PCA                 | ⏳ Planned  |
| K-Means             | ⏳ Planned  |

---

## Deep Learning

| Algorithm   | Status     |
| ----------- | ---------- |
| Perceptron  | ⏳ Planned |
| MLP         | ⏳ Planned |
| CNN         | ⏳ Planned |
| RNN         | ⏳ Planned |
| LSTM        | ⏳ Planned |
| GRU         | ⏳ Planned |
| Transformer | ⏳ Planned |
| Attention   | ⏳ Planned |

---

# Common Components

Many implementations reuse components developed from scratch.

| Component            | Status |
| -------------------- | ------ |
| SGD Optimizer        | ✅     |
| Data Loader          | ✅     |
| Loss Functions       | 🔄     |
| Activation Functions | ⏳     |
| Metrics              | ⏳     |
| Initializers         | ⏳     |

---

# Design Philosophy

Each implementation follows the same principles:

- No Scikit-Learn implementation
- No pre-built optimizers
- Mathematical equations translated directly into code
- Modular implementation
- Well documented
- Reproducible experiments

---

# What "From Scratch" Means

Algorithms may use:

- Python
- NumPy
- PyTorch tensors for numerical computation

Algorithms do **not** use:

- sklearn models
- torch.optim
- torch.nn modules
- pretrained implementations

Every optimization step, forward pass, and training loop is implemented manually.

---

# Progress

Current Progress

- Algorithms Completed: **1**
- Utility Modules: **2**
- Total Lines of Code: Growing...

---

# Future Work

- Automatic differentiation
- GPU support
- More optimizers (Momentum, RMSProp, Adam)
- Better visualization
- Model serialization
- Benchmark against Scikit-Learn

---

# References

- Pattern Recognition and Machine Learning — Bishop
- The Elements of Statistical Learning
- Deep Learning — Goodfellow
- Dive into Deep Learning (D2L)
- Stanford CS229
