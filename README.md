This repository contains the code and materials for a tutorial that investigates how L1, L2, and Dropout regularization affect the behaviour, training dynamics, and generalization performance of a multilayer perceptron (MLP) trained on the MNIST handwritten digit dataset. The project explains the theoretical motivations behind regularization, demonstrates model implementation, and presents experimental results including learning curves, accuracy metrics, and confusion matrices. The aim is to help learners understand how regularization prevents overfitting and improves model stability.

Contents
tutorial.pdf – Final tutorial document
regularization_experiments.ipynb – Jupyter Notebook with all training, evaluation, and visualisation code
figures/ – Folder for generated plots such as learning curves, accuracy comparison charts, and confusion matrices
README.md – This file
LICENSE – License information (MIT recommended)

Dataset
MNIST is used as the dataset for all experiments. It contains 70,000 grayscale handwritten digit images (60,000 for training and 10,000 for testing). Each image is 28×28 pixels, flattened into 784 features. There are 10 balanced classes (digits 0–9). Preprocessing includes normalizing pixel intensities to the range 0–1 and using a 20% validation split taken from the training data. MNIST loads directly from tensorflow.keras.datasets.

How to Run the Notebook

Install the required Python libraries: tensorflow, numpy, matplotlib, scikit-learn
Command: pip install tensorflow numpy matplotlib scikit-learn

Open the notebook using Jupyter: jupyter notebook regularization_experiments.ipynb

Running the notebook will:
– Train four MLP models: Baseline, L1, L2, and Dropout
– Generate training and validation learning curves
– Produce bar charts comparing test accuracy across models
– Output confusion matrices for each model
– Compute classification metrics including accuracy, precision, recall, and F1-scores

All tutorial figures can be reproduced directly from the notebook.

Repository Structure
regularization_experiments.ipynb
tutorial.pdf
figures/
loss_curves.png
accuracy_curves.png
test_accuracy_bar.png
confusion_matrix_baseline.png
confusion_matrix_l1.png
confusion_matrix_l2.png
confusion_matrix_dropout.png
README.md
LICENSE

Accessibility
This tutorial and repository follow accessibility best practices:
– All plots use colour-blind-safe palettes
– High-contrast text and axes are used
– Alt-text descriptions are provided for every figure in the tutorial
– Tables are designed to remain readable without relying on colours
– Headings follow a consistent hierarchical structure to support screen-reader navigation

References
Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
Tibshirani (1996). Regression Shrinkage and Selection via the Lasso.
Bishop (1995). Training with Noise is Equivalent to Tikhonov Regularization.
Goodfellow, Bengio & Courville (2016). Deep Learning. MIT Press.

License
This project is distributed under the MIT License. Users are free to reuse and modify the code according to the terms described in the LICENSE file.
