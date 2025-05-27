# K-Nearest Neighbors (KNN) Classification Projects

This repository contains three classification projects using the K-Nearest Neighbors (KNN) algorithm. The goal is to train a KNN model on different datasets and evaluate its performance using various metrics.

## 🔍 Projects Overview

### 1. Iris Dataset
- **Source:** `sklearn.datasets`
- **Description:** A classic dataset for classifying iris flowers into three species: Setosa, Versicolor, and Virginica based on features like petal and sepal length/width.
- **Steps:**
  - Load dataset using `sklearn.datasets`
  - Split into training and testing sets
  - Train KNN model
  - Make predictions
  - Evaluate using Confusion Matrix, R² Score, and Classification Report

### 2. Wine Dataset
- **Source:** `sklearn.datasets`
- **Description:** Wine classification into three classes based on chemical analysis of wines.
- **Steps:**
  - Load dataset using `sklearn.datasets`
  - Preprocess the data (if needed)
  - Train the KNN model
  - Predict test and train data
  - Evaluate model performance

### 3. Breast Cancer Dataset
- **Source:** [Kaggle Dataset]([https://www.kaggle.com/](https://www.kaggle.com/datasets/gkalpolukcu/knn-algorithm-dataset/code))
- **Description:** Predict whether a tumor is benign or malignant based on various cell features.
- **Steps:**
  - Load dataset (already cleaned)
  - Drop unnecessary columns
  - Train-test split
  - Train KNN model
  - Predict on both training and testing sets
  - Compare predictions
  - Evaluate using:
    - Confusion Matrix
    - R² Score
    - Classification Report

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (optional for visualizations)
- Jupyter Notebook / VSCode

---

## 📊 Evaluation Metrics

- **Confusion Matrix**: Shows true positives, false positives, true negatives, and false negatives.
- **R² Score**: Used to measure the goodness of fit (mostly for regression but used here for additional comparison).
- **Classification Report**: Includes precision, recall, f1-score, and support.

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Talalzahid868/Machine-Learning-Projects.git
   cd Machine-Learning-Projects/KNN_Practice

