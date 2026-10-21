"""Compute common binary-classification metrics using NumPy."""
import numpy as np


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length.")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Labels must be binary: 0 or 1.")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classification_metrics(y_true, y_pred):
    counts = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = (counts[k] for k in ("TP", "TN", "FP", "FN"))
    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # Example only: replace these arrays with predictions from your own model.
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

    results = classification_metrics(y_true, y_pred)
    print("Confusion counts")
    print(f"TP={results['TP']}  TN={results['TN']}  FP={results['FP']}  FN={results['FN']}")
    print("\nClassification metrics")
    for metric in ("accuracy", "precision", "recall", "f1"):
        print(f"{metric.capitalize():<10}: {results[metric]:.3f}")

    print("\nReminder: for imbalanced data, inspect precision, recall, F1, and class counts—not accuracy alone.")


if __name__ == "__main__":
    main()
