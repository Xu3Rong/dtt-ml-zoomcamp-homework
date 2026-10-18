# Machine Learning Zoomcamp 2026 — Week 4: Model Evaluation Metrics

This folder contains the complete evaluation pipeline for Week 4, covering ROC AUC feature ranking, threshold tuning across precision, recall, and F1 score, and 5-fold cross-validation for hyperparameter regularization.

---

## Summary of Verified Answers

| Question | Topic / Task | Selected Answer | Value / Metric |
|---|---|---|---|
| **Q1** | Numerical feature with highest ROC AUC | **`lead_score`** | AUC = `0.7882` |
| **Q2** | Validation ROC AUC (Logistic Regression) | **`0.732`** | Raw = `0.7318` |
| **Q3** | Precision & recall intersection threshold | **`0.63`** | Diff = `0.0075` |
| **Q4** | Threshold where F1 score is maximal | **`0.41`** | Max F1 = `0.7576` |
| **Q5** | Standard deviation across 5 folds ($C=1.0$) | **`0.007`** | Std = `0.0069` |
| **Q6** | Best regularizer $C$ via 5-fold cross-validation | **`0.001`** | Mean AUC = `0.749` |

---

## Key Takeaways

1. **Feature Evaluation via AUC:** Evaluated single-variable discriminative power without requiring feature scaling or linear assumptions.
2. **Threshold Decisions:** A decision boundary around `0.41` maximizes the balanced $F_1$ score, whereas `0.63` balances precision and recall symmetrically.
3. **Cross-Validation Stability:** 5-fold cross-validation demonstrated low score variance ($\sigma \approx 0.007$), confirming model stability across different splits.