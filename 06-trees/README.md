# Module 6: Decision Trees and Ensemble Learning (Machine Learning Zoomcamp 2026)

This repository contains the end-to-end model training, hyperparameter tuning, and ensemble evaluation pipeline for Module 6 of the DataTalksClub Machine Learning Zoomcamp (2026 Cohort). The goal is to build and tune regression models predicting vehicle fuel efficiency (`fuel_efficiency_mpg`) using decision trees, random forests, and gradient-boosted trees (XGBoost).

---

## 📋 Homework 6 Verified Answers

| Question | Topic | Selected Option | Computed / Validation Metric |
|---|---|---|---|
| **Question 1** | Decision tree root split (`max_depth=1`) | **`model_year`** | Split rule: `model_year <= 1997.50` |
| **Question 2** | Random forest baseline RMSE (`n=10`) | **`1.837`** | Validation RMSE: `1.8332` |
| **Question 3** | Lowest RMSE across `n_estimators` | **`150`** | `1.769` (RMSE: `1.7692`) |
| **Question 4** | Optimal `max_depth` (lowest mean RMSE) | **`10`** | Mean RMSE: `1.7544` |
| **Question 5** | Most important feature (`max_depth=20`) | **`vehicle_weight`** | Impurity reduction score: `0.2033` |
| **Question 6** | Best XGBoost learning rate (`eta`) | **`0.1`** | Final validation RMSE: `1.7248` (vs. `1.8265`) |

---

## 📂 Project Structure

```text
06-trees/
├── car_fuel_efficiency_2026.csv  # Pinned cohort dataset (10,000 rows)
├── homework6.ipynb               # Complete exploratory analysis, tree plots, and tuning notebook
└── README.md                     # Documentation, verified answer key, and reproduction guide
```

---

## ⚙️ Dataset & Preprocessing Pipeline

* **Dataset:** Pinned 2026 release (`car_fuel_efficiency_2026.csv`).
* **Target Variable:** `fuel_efficiency_mpg` (miles per gallon continuous target).
* **Missing Value Strategy:** Zero-imputation (`df.fillna(0)`).
* **Partition Strategy:**
  1. Full Train / Test split: `test_size=0.2`, `random_state=1`
  2. Train / Validation split: `test_size=0.25`, `random_state=1`
  3. Resulting split sizes: 
     * **Train:** 6,000 records (60%)
     * **Validation:** 2,000 records (20%)
     * **Test:** 2,000 records (20%)
* **Feature Encoding:** `DictVectorizer(sparse=True)` generating a 16-feature encoded matrix.

---

## 🔬 Experimentation & Findings

### 1. Decision Tree Split Boundary (Question 1)
A single-split regressor (`DecisionTreeRegressor(max_depth=1)`) evaluated the optimal initial partition:
```text
|--- model_year <= 1997.50
|   |--- value: [28.61]
|--- model_year >  1997.50
|   |--- value: [31.17]
```
The tree selected **`model_year`** as the feature providing the maximum variance reduction at the root.

### 2. Random Forest Baseline & Convergence (Questions 2 & 3)
Training `RandomForestRegressor(random_state=1, n_jobs=-1)` across tree counts demonstrated expected error decay:

| `n_estimators` | Validation RMSE | Rounded (3 Decimals) |
|---|---|---|
| **10** | 1.8332 | 1.833 |
| **50** | 1.7740 | 1.774 |
| **100** | 1.7698 | 1.770 |
| **150** | **1.7692** | **1.769** |

* Baseline RMSE matches option **`1.837`** (Question 2).
* Error monotonically decreases and stabilizes at **`n_estimators=150`** (Question 3).

### 3. Depth Regularization Tuning (Question 4)
Evaluating `max_depth` across all tree counts (`[10, 50, 100, 150]`) demonstrated that shallower trees mitigate overfitting:

| `max_depth` | Mean Validation RMSE |
|---|---|
| **10** | **1.7544** |
| **15** | 1.7842 |
| **20** | 1.7868 |
| **25** | 1.7863 |

Restricting `max_depth=10` yields the lowest mean validation RMSE.

### 4. Feature Importance Extraction (Question 5)
Extracting `feature_importances_` from `RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1)` reveals the relative Gini/MSE gain:

| Feature | Importance Score | Rank Among Candidates |
|---|---|---|
| **`vehicle_weight`** | **0.203313** | **1** |
| `horsepower` | 0.078271 | 2 |
| `engine_displacement` | 0.061281 | 3 |
| `acceleration` | 0.054998 | 4 |

`vehicle_weight` is the single most dominant predictor of fuel efficiency.

### 5. Gradient Boosted Trees with XGBoost (Question 6)
Two 100-round gradient boosting models were trained on `xgb.DMatrix` with fixed parameters (`max_depth=6`, `min_child_weight=1`, `seed=1`):

* **`eta=0.3`:** Final Validation RMSE = `1.8265`
* **`eta=0.1`:** Final Validation RMSE = **`1.7248`**

A smaller learning rate (`eta=0.1`) avoids premature overshooting and generalizes significantly better across 100 boosting rounds.

---

## 🚀 Reproduction Guide

### 1. Environment Setup

Ensure the required dependencies are installed:

```bash
pip install pandas numpy scikit-learn xgboost
```

### 2. Run Analysis

Launch Jupyter Notebook and execute `homework6.ipynb`:

```bash
jupyter notebook homework6.ipynb
```