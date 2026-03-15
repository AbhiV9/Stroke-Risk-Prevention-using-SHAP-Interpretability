# Stroke Risk Prediction Using Machine Learning and SHAP Interpretability

## Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of individuals at high risk can significantly improve prevention and treatment outcomes.

This project develops a machine learning pipeline that predicts stroke risk using patient health data. The workflow includes data preprocessing, handling class imbalance, training several machine learning models, evaluating their performance, and preparing the model for interpretability using **SHAP (SHapley Additive exPlanations)**.

> The primary goal is not only to build a predictive model, but to understand **which health factors contribute most strongly to stroke risk**.

---

## Dataset

The **Healthcare Stroke Dataset** contains demographic and medical information used to predict whether a patient experienced a stroke.

| Feature | Description |
|---|---|
| `gender` | Gender of the patient |
| `age` | Patient's age |
| `hypertension` | Whether the patient has hypertension |
| `heart_disease` | Whether the patient has heart disease |
| `ever_married` | Marital status |
| `work_type` | Type of employment |
| `Residence_type` | Urban or rural residence |
| `avg_glucose_level` | Average blood glucose level |
| `bmi` | Body Mass Index |
| `smoking_status` | Smoking history |
| `stroke` | **Target variable** — whether the patient had a stroke |

> ⚠️ **Class Imbalance:** Only ~5% of patients experienced a stroke. Special techniques are required to handle this imbalance.

---

## Project Workflow

Data Loading → EDA → Data Cleaning → Feature Engineering → Visualization → Train/Test Split → Feature Scaling → Handle Imbalance → Model Training → Evaluation → Threshold Optimization → Precision-Recall Analysis

### 1. Data Loading
The dataset is loaded using **Pandas**. The non-informative `id` column is removed to prevent noise in the model.

### 2. Exploratory Data Analysis
- Inspection of data types
- Missing value detection
- Summary statistics
- Distribution analysis

### 3. Data Cleaning
Missing values in the `bmi` column are imputed using **median imputation**, which is less sensitive to outliers than the mean.

### 4. Feature Engineering
Categorical variables are transformed using multiple encoding strategies:

- **Binary Encoding** — for variables like `gender`, `ever_married`, and `Residence_type`
- **Rare Category Handling** — the rare `"Other"` gender category is replaced with the dominant class to avoid training instability
- **One-Hot Encoding** — for multi-class variables like `work_type` and `smoking_status` (with dummy variable trap prevention)

### 5. Data Visualization
Key visualizations include:
- Stroke distribution plot
- Age distribution grouped by stroke occurrence
- Correlation heatmap

**Insights revealed:**
- Stroke risk increases significantly with age
- Hypertension and heart disease are correlated with stroke risk
- Higher glucose levels are associated with increased stroke probability

### 6. Train/Test Split
Stratified splitting ensures both sets maintain the same stroke-to-non-stroke ratio as the original dataset.

### 7. Feature Scaling
**Standardization** (mean = 0, std = 1) is applied to numerical features so that no variable dominates due to scale differences.

### 8. Handling Class Imbalance
**SMOTE (Synthetic Minority Oversampling Technique)** is applied to generate synthetic minority-class examples, improving the model's ability to detect stroke cases.

---

## Models Trained

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline model — simple, interpretable, commonly used in medical prediction |
| **Decision Tree** | Rule-based model capable of capturing nonlinear relationships |
| **Random Forest** | Ensemble method combining multiple decision trees; reduces overfitting |

All models are trained with **balanced class weights** to further penalize misclassification of stroke cases.

---

## Model Evaluation

Models are evaluated using:

- **Confusion Matrix** — True/false positives and negatives
- **Precision** — Of predicted strokes, how many were actual strokes
- **Recall** — Of actual strokes, how many were correctly detected
- **F1 Score** — Harmonic mean of precision and recall
- **Precision-Recall Curves** — Better suited than accuracy for imbalanced datasets

> In healthcare, **recall is prioritized** — missing a real stroke case is far more costly than a false positive.

### Threshold Optimization
The default classification threshold (0.5) is lowered to **increase recall**, ensuring more potential stroke cases are flagged even at the cost of additional false positives.

---

## Technologies Used

- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical computation
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Model training and evaluation
- **Imbalanced-learn** — SMOTE implementation
- **SHAP** — Model interpretability *(in progress)*

---

## Future Work

- [ ] **SHAP Interpretability** — Identify features most strongly influencing stroke predictions and generate individual-level explanations
- [ ] **Advanced Models** — Experiment with gradient boosting methods (XGBoost, LightGBM)
- [ ] **Hyperparameter Tuning** — Grid search / randomized search optimization
- [ ] **Cross Validation** — Improve reliability of performance estimates

---

## Motivation

This project explores how machine learning can assist with healthcare risk prediction. The focus is on building models that are both **accurate and interpretable** — because in medical applications, understanding *why* a model makes a prediction is just as important as the prediction itself.

By combining ML with interpretability techniques like SHAP, this project aims to contribute toward more **transparent and trustworthy AI in healthcare**.
