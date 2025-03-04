# Kidney Disease Prediction using Machine Learning

This repository contains a Jupyter Notebook that implements **Machine Learning models** to predict the likelihood of a patient having kidney disease based on their health records. The dataset includes clinical parameters such as age, blood pressure, glucose levels, and other medical attributes.

## 📌 Features
- **Data Preprocessing**
  - Handles missing values (using imputation techniques)
  - Encodes categorical variables
  - Applies Min-Max scaling for numerical features
  
- **Class Balancing**
  - Uses **SMOTE (Synthetic Minority Over-sampling Technique)** to handle dataset imbalance

- **Model Training and Evaluation**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
  - **Decision Tree Regressor**
  - **Random Forest Classifier**
  - Evaluates models using **Accuracy, Confusion Matrix, Precision, Recall, and F1-score**

## 📂 Files
- `kidney disease FINAL.ipynb` - Jupyter Notebook containing all implementations
- `kidney_disease.csv` - Dataset containing patient health records (ensure it is placed in the working directory)

## 🛠 Dependencies
The notebook uses the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`
- `joblib`

To install missing dependencies, run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

## 🔍 How to Use
1. Open the Jupyter Notebook (`kidney disease FINAL.ipynb`).
2. Run each cell sequentially to preprocess data, balance classes, train models, and evaluate performance.
3. Modify parameters (e.g., test size, solver, scaling method) to experiment with different setups.

## 📈 Results
- **Logistic Regression** provides a baseline performance.
- **SVM** with optimized hyperparameters improves prediction accuracy.
- **Decision Tree** shows moderate performance.
- **Random Forest Classifier** achieves the highest accuracy and balanced precision-recall.
