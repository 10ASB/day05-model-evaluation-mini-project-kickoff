# Day 5 — Model Evaluation & Mini-Project 1 Kickoff

**AI/ML Internship · Week 1 · Day 5**

Today focuses on evaluating classification models properly and planning Mini-Project 1.

## Learning goals

- Read a confusion matrix
- Calculate accuracy, precision, recall, and F1-score
- Understand why accuracy alone can be misleading for imbalanced classes
- Compare training and test performance to spot possible overfitting
- Define a mini-project problem, target, metrics, and baseline

## Files

| File | Purpose |
|---|---|
| `model_evaluation.py` | Implement classification metrics from scratch with NumPy |
| `mini_project_plan.md` | Starter template for the first mini-project |

## Requirements and run

```bash
python -m pip install numpy
python model_evaluation.py
```

## Metric definitions

For binary classification:

- **TP**: predicted positive and actually positive
- **TN**: predicted negative and actually negative
- **FP**: predicted positive but actually negative
- **FN**: predicted negative but actually positive

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

When a denominator is zero, this educational implementation returns 0 for that metric.

## Evaluation checklist

1. Keep a held-out test set separate from training.
2. Choose metrics that fit the task and class balance.
3. Compare training and test performance.
4. Inspect errors, not only the final score.
5. Avoid data leakage: fit preprocessing only on training data.

## Mini-project kickoff

Use `mini_project_plan.md` to record the problem, data source, target, baseline, metrics, and planned deliverables. Do not claim a model result until you have run and documented the experiment.

---

## Roadmap description

### Theory
- Bias–variance tradeoff; overfitting vs underfitting, using learning curves.
- Cross-validation: k-fold, stratified k-fold, and time-series split.
- Classification metrics: confusion matrix, accuracy, precision, recall, F1, ROC-AUC, and PR-AUC — including when accuracy can mislead.
- Regression metrics: MAE, MSE, RMSE, R², and MAPE.
- Class imbalance: class weights, SMOTE, and threshold tuning.
- Choose an evaluation metric based on the business problem.

### Hands-on lab
- Build a reusable `evaluate_model()` utility with metrics, a confusion matrix, and ROC/PR curves.
- Re-evaluate Day 4 models properly using stratified cross-validation.
- Begin Mini-Project 1, using an individual dataset per intern (see Section 7 of the internship roadmap).

### Week 1 review
- The roadmap schedules a Week 1 demo and review from **4:00–5:00 PM**. Each intern presents for 10 minutes with slides.

### Expected deliverable
- A reusable evaluation utility, documented evaluation results for the Day 4 models, and a Mini-Project 1 plan/initial work.

### Practical completion checklist
- [ ] Explain bias, variance, overfitting, and underfitting.
- [ ] Compare k-fold, stratified k-fold, and time-series validation.
- [ ] Calculate classification and regression metrics.
- [ ] Explain why accuracy alone may be misleading for imbalanced classes.
- [ ] Implement `evaluate_model()` and include confusion-matrix and ROC/PR-curve support.
- [ ] Re-evaluate Day 4 models with stratified cross-validation.
- [ ] Record the mini-project problem, dataset, baseline, metric, and next steps.
