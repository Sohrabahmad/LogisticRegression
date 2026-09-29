# ❤️ Heart Disease Prediction using Logistic Regression

A machine learning project that uses **Logistic Regression** to predict the presence of heart disease based on patient health and clinical attributes.

The project covers data loading, feature-target separation, train-test splitting, Logistic Regression, classification metrics, confusion matrix analysis, and feature scaling using `StandardScaler`.

---

## 📌 Project Overview

The objective of this project is to build a binary classification model that predicts whether a patient belongs to:

* `0` → No heart disease
* `1` → Heart disease

The model is trained using patient-related features such as age, sex, chest pain type, blood pressure, cholesterol, maximum heart rate, and other clinical measurements.

---

## 📂 Project Structure

```text
Heart-Disease-Logistic-Regression/
│
├── heart(1).csv
├── logistic_regression(1).ipynb
└── README.md
```

---

## 📊 Dataset

The `heart(1).csv` dataset contains:

* **303 records**
* **14 columns**
* **13 input features**
* **1 target variable**

### Dataset Features

| Feature    | Description                           |
| ---------- | ------------------------------------- |
| `age`      | Age of the patient                    |
| `sex`      | Sex of the patient                    |
| `cp`       | Chest pain type                       |
| `trestbps` | Resting blood pressure                |
| `chol`     | Serum cholesterol                     |
| `fbs`      | Fasting blood sugar                   |
| `restecg`  | Resting electrocardiographic results  |
| `thalach`  | Maximum heart rate achieved           |
| `exang`    | Exercise-induced angina               |
| `oldpeak`  | ST depression induced by exercise     |
| `slope`    | Slope of the peak exercise ST segment |
| `ca`       | Number of major vessels               |
| `thal`     | Thalassemia-related value             |
| `target`   | Heart disease target                  |

### Target

```text
0 → No heart disease
1 → Heart disease
```

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Seaborn**
* **Scikit-learn**
* **Jupyter Notebook**

### Main Libraries

```python
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
```

---

# 🔍 Data Preparation

## 1. Load Dataset

The dataset is loaded using Pandas:

```python
heart_data = pd.read_csv("heart.csv")
```

The notebook explores the dataset using:

```python
heart_data.head()
```

---

## 2. Separate Features and Target

The target column is separated from the input features:

```python
X = heart_data.drop("target", axis=1)
y = heart_data["target"]
```

Where:

* `X` → Input features
* `y` → Target variable

---

# ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split

| Dataset  | Records |
| -------- | ------: |
| Training |     242 |
| Testing  |      61 |

---

# 🤖 Logistic Regression

A Logistic Regression classifier is created with:

```python
model = LogisticRegression(max_iter=10000)
```

The model is trained using:

```python
model.fit(X_train, y_train)
```

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

---

# 📏 Model Evaluation

The project evaluates the classifier using several classification metrics.

## Accuracy

Accuracy represents the proportion of correctly classified samples.

```python
accuracy_score(y_test, y_pred)
```

## Precision

Precision measures how many predicted positive cases were actually positive.

```python
precision_score(y_test, y_pred)
```

## Recall

Recall measures how many actual positive cases were correctly identified.

```python
recall_score(y_test, y_pred)
```

## F1 Score

F1 score combines precision and recall into a single metric.

```python
f1_score(y_test, y_pred)
```

---

# 📊 Initial Model Results

The notebook reports the following results for the initial Logistic Regression model:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **88.52%** |
| Precision | **87.88%** |
| Recall    | **90.63%** |
| F1 Score  | **89.23%** |

The model achieved an accuracy of approximately **88.52%** on the test set.

---

# 🔲 Confusion Matrix

The notebook generates the following confusion matrix:

```text
[[25  4]
 [ 3 29]]
```

It can be interpreted as:

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------: | ----------: |
| **Actual 0** |          25 |           4 |
| **Actual 1** |           3 |          29 |

This allows the model's classification errors and correct predictions to be examined.

---

# 📋 Classification Report

The notebook produces the following classification performance:

```text
              precision    recall  f1-score   support

           0       0.89      0.86      0.88        29
           1       0.88      0.91      0.89        32

    accuracy                           0.89        61
   macro avg       0.89      0.88      0.88        61
weighted avg       0.89      0.89      0.89        61
```

---

# ⚖️ Feature Scaling

The notebook also applies **StandardScaler** to standardize the input features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to both training and testing data.

This produces standardized features with comparable scales.

---

# 🔄 Logistic Regression After Scaling

After applying feature scaling, Logistic Regression is trained again:

```python
model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

The notebook reports:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **85.25%** |
| Precision | **87.10%** |

The notebook therefore provides a direct comparison between the model before and after feature scaling.

---

# 🔄 Machine Learning Workflow

```text
Load Heart Disease Dataset
          ↓
Explore Dataset
          ↓
Separate Features & Target
          ↓
Train-Test Split
          ↓
Train Logistic Regression
          ↓
Generate Predictions
          ↓
Calculate Classification Metrics
          ↓
Create Confusion Matrix
          ↓
Generate Classification Report
          ↓
Apply StandardScaler
          ↓
Train Logistic Regression Again
          ↓
Compare Results
```

---

# 🎯 Learning Objectives

This project provides practical experience with:

* Binary classification
* Logistic Regression
* Pandas data handling
* Train-test splitting
* Feature-target separation
* Classification metrics
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report
* Feature scaling
* StandardScaler
* Model comparison

---

# 💡 Key Takeaways

* Logistic Regression can be used for binary classification problems.
* The dataset contains 13 input features and one binary target.
* The initial model achieved **88.52% test accuracy**.
* Precision, recall, and F1 score provide additional information beyond accuracy.
* A confusion matrix helps understand correct and incorrect classifications.
* Feature scaling was also tested using `StandardScaler`.
* In this notebook, the scaled model produced **85.25% accuracy**, compared with **88.52%** for the initial model.

---

# 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Heart-Disease-Logistic-Regression
```

### 2. Install Dependencies

```bash
pip install pandas seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the Notebook

```text
logistic_regression(1).ipynb
```

Make sure the dataset is available in the same project directory and update the filename in the notebook if necessary.

---

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech — Computer Science Engineering (AI/ML)

---

## ⭐ Project Purpose

This project is part of a hands-on learning journey in **Machine Learning, Data Analysis, and AI/ML**, with a focus on understanding Logistic Regression and classification model evaluation.

If you find this project useful, consider giving the repository a ⭐.
