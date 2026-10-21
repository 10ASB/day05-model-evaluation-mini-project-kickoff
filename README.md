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
