# Heart Disease Prediction using Machine Learning

## Overview

This repository contains a machine learning pipeline built for Task 4: Disease Prediction from Medical Data, part of the CodeAlpha Machine Learning Internship.

The goal is to classify whether a patient has heart disease based on clinical and diagnostic indicators, using an **XGBoost Classifier** (Gradient Boosting).

---

## Dataset Details

The dataset used is the standard UCI Heart Disease Dataset. It consists of patient records with the following feature groups:

- **Demographics:** `age`, `sex`
- **Clinical Indicators:** `cp` (chest pain type), `trestbps` (resting blood pressure)
- **Electrocardiographic & Stress Results:** `restecg`, `thalach` (max heart rate)
- **Diagnostic Tests:** `ca` (number of major vessels), `thal` (thalassemia type)
- **Target:** `target` (`0` = No Heart Disease, `1` = Heart Disease)

---

## Project Structure & Architecture

The code is structured modularly with clear function pipelines:

- **`load_and_preprocess_heart_data()`** — Handles dataset loading, feature-target separation, and preprocessing
- **`evaluate_classifier()`** — Computes accuracy, ROC-AUC, and classification metrics (Precision, Recall, F1-Score)

---

## Model Approach

### XGBoost Classifier

- **Mechanism:** Sequential gradient boosting method
- **Bias-Variance Profile:** Primarily targets bias reduction by iteratively learning from prior trees' errors, making it well-suited for capturing subtle patterns in clinical data

---

## Installation & Running the Script

### Prerequisites

Ensure Python 3.x and the required dependencies are installed:

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

### Execution

Run the script inside VS Code:

```bash
python xgboost_prediction.py
```

---

## Author & Acknowledgments

- **Developer:** Wireko Fosu Eric
- **Program:** Machine Learning Internship at CodeAlpha
