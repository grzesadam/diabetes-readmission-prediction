"""Reusable model evaluation helpers for binary readmission models."""

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


def _positive_class_scores(fitted_model, X):
    """Return continuous positive-class scores and whether they are probabilities.

    Parameters
    ----------
    fitted_model : estimator
        Fitted classifier or search object that may expose ``predict_proba`` or
        ``decision_function``.
    X : array-like
        Feature matrix to score.

    Returns
    -------
    tuple
        Pair of ``(scores, are_probabilities)``. ``scores`` is ``None`` when
        the estimator does not expose continuous class scores.
    """
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1], True
    if hasattr(fitted_model, "decision_function"):
        return fitted_model.decision_function(X), False
    return None, False


def _plot_confusion_matrix(y_true, y_pred, title, class_labels):
    """Plot a labeled confusion matrix and return its raw counts.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Predicted binary labels.
    title : str
        Plot title.
    class_labels : sequence of str
        Display labels for the negative and positive classes.

    Returns
    -------
    numpy.ndarray
        Two-by-two confusion matrix ordered by the labels found in ``y_true``
        and ``y_pred``.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()
    return cm


def _plot_roc_auc_with_thresholds(
    y_true,
    y_score,
    model_name,
    threshold_values=None,
):
    """Plot an ROC curve with selected probability thresholds highlighted.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_score : array-like
        Continuous positive-class scores, usually probabilities.
    model_name : str
        Name included in the plot title.
    threshold_values : array-like, optional
        Thresholds to annotate on the ROC curve. Defaults to values from 0.40
        to 0.59 in increments of 0.01.

    Returns
    -------
    tuple
        Pair of ``(roc_auc, threshold_points)`` where ``threshold_points`` is a
        DataFrame with threshold, false-positive-rate, and true-positive-rate
        columns.
    """
    if threshold_values is None:
        threshold_values = np.arange(0.4, 0.6, 0.02)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    threshold_points = []
    for threshold_value in threshold_values:
        y_pred_at_threshold = (y_score >= threshold_value).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_at_threshold).ravel()
        threshold_points.append(
            {
                "threshold": threshold_value,
                "fpr": fp / (fp + tn),
                "tpr": tp / (tp + fn),
            }
        )

    threshold_points = pd.DataFrame(threshold_points)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="lightgray", linewidth=2, label=f"Full ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
    plt.plot(
        threshold_points["fpr"],
        threshold_points["tpr"],
        marker="o",
        color="tab:blue",
        linewidth=2,
        label=f"Thresholds {threshold_values.min():.2f}-{threshold_values.max():.2f}",
    )

    for _, row in threshold_points.iterrows():
        plt.annotate(
            f"{row['threshold']:.2f}",
            (row["fpr"], row["tpr"]),
            textcoords="offset points",
            xytext=(15, -5),
            fontsize=8,
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve with Thresholds Highlighted: {model_name}")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    display(threshold_points)
    return roc_auc, threshold_points


def _threshold_scores(y_true, y_score, threshold_values):
    """Return classification metrics across candidate thresholds."""
    threshold_scores = []
    for threshold_value in threshold_values:
        y_pred_at_threshold = (y_score >= threshold_value).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_at_threshold).ravel()
        specificity = tn / (tn + fp)
        recall = recall_score(y_true, y_pred_at_threshold, zero_division=0)
        threshold_scores.append(
            {
                "threshold": threshold_value,
                "f1": f1_score(y_true, y_pred_at_threshold, zero_division=0),
                "precision": precision_score(y_true, y_pred_at_threshold, zero_division=0),
                "recall": recall,
                "specificity": specificity,
                "balanced_accuracy": (recall + specificity) / 2,
            }
        )
    return pd.DataFrame(threshold_scores)


def _best_threshold(y_true, y_score, threshold_values, metric):
    """Return the threshold with the best selected threshold metric."""
    threshold_scores = _threshold_scores(y_true, y_score, threshold_values)
    sort_columns = {
        "f1": ["f1", "recall", "precision"],
        "balanced_accuracy": ["balanced_accuracy", "f1", "recall"],
    }[metric]
    best_row = threshold_scores.sort_values(
        sort_columns,
        ascending=False,
    ).iloc[0]
    return float(best_row["threshold"]), threshold_scores


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    model_name="Model",
    param_grid=None,
    scoring=None,
    cv=5,
    n_jobs=-1,
    threshold=0.50,
    threshold_validation_size=0.20,
    threshold_random_state=42,
    plot_confusion_matrix=True,
    plot_roc_auc=True,
    threshold_values=None,
    class_labels=("No Readmission", "Readmission"),
):
    """Fit a classifier, optionally tune it, and print/plot standard evaluation outputs.

    By default, GridSearchCV uses average precision, which is better suited to
    imbalanced binary classification than threshold-dependent accuracy. The
    ROC-AUC chart is optional because it requires continuous scores from
    predict_proba or decision_function.

    Parameters
    ----------
    model : estimator or sklearn.pipeline.Pipeline
        Classifier to fit. Non-pipeline estimators are wrapped in a single-step
        pipeline named ``"model"`` so parameter grids can target that step.
    X_train, X_test : array-like
        Training and test feature matrices.
    y_train, y_test : array-like
        Training and test labels for the binary readmission outcome.
    model_name : str, default="Model"
        Name used in plot titles and the returned result dictionary.
    param_grid : dict or list of dict, optional
        Parameter grid passed to ``GridSearchCV``. When omitted, the model is fit
        directly without cross-validation tuning.
    scoring : str, callable, or scorer, optional
        Scoring strategy for ``GridSearchCV``. Defaults to average precision.
    cv : int or cross-validation splitter, default=5
        Cross-validation strategy passed to ``GridSearchCV``. Integer values
        are converted to ``StratifiedKFold`` with shuffling so each fold keeps
        a similar readmission positive rate.
    n_jobs : int, default=-1
        Number of parallel jobs used by ``GridSearchCV``.
    threshold : float, "best_f1", or "best_balanced_accuracy", default=0.50
        Probability threshold used to convert positive-class probabilities into
        binary predictions when ``predict_proba`` is available. If set to
        ``"best_f1"``, the threshold is selected from ``threshold_values`` by
        maximizing positive-class F1 on a validation slice of the training set,
        then applied once to the test set. If set to
        ``"best_balanced_accuracy"``, the threshold is selected by maximizing
        the average of recall and specificity.
    threshold_validation_size : float, default=0.20
        Fraction of the provided training data reserved for threshold selection
        when ``threshold`` is a threshold-search mode. The model and any grid
        search are fit on the remaining training rows.
    threshold_random_state : int, default=42
        Random seed used for the threshold-selection validation split.
    plot_confusion_matrix : bool, default=True
        Whether to show the confusion-matrix heatmap.
    plot_roc_auc : bool, default=True
        Whether to plot the ROC-AUC curve when continuous scores are available.
    threshold_values : array-like, optional
        Thresholds to annotate on the ROC curve.
    class_labels : tuple of str, default=("No Readmission", "Readmission")
        Axis labels used in the confusion matrix.

    Returns
    -------
    dict
        Evaluation artifacts including the fitted estimator, predictions,
        optional continuous scores, confusion matrix, scalar metrics, ROC-AUC,
        and annotated threshold points.
    """
    if threshold_values is None:
        threshold_values = np.arange(0.4, 0.6, 0.01)

    scoring = scoring or "average_precision"
    y_train_1d = np.asarray(y_train).ravel()

    pipeline = model if isinstance(model, Pipeline) else Pipeline([("model", model)])

    selected_threshold = threshold
    threshold_selection_scores = None
    threshold_selection_source = None
    threshold_metric_map = {
        "best_f1": "f1",
        "best_balanced_accuracy": "balanced_accuracy",
    }
    threshold_mode = str(threshold) if isinstance(threshold, str) else None
    if threshold_mode is not None and threshold_mode not in threshold_metric_map:
        raise ValueError(
            "Unknown threshold mode. Use a numeric threshold, 'best_f1', "
            "or 'best_balanced_accuracy'."
        )

    X_fit = X_train
    y_fit = y_train_1d
    X_threshold = None
    y_threshold = None

    if threshold_mode in threshold_metric_map:
        X_fit, X_threshold, y_fit, y_threshold = train_test_split(
            X_train,
            y_train_1d,
            test_size=threshold_validation_size,
            random_state=threshold_random_state,
            stratify=y_train_1d,
        )
        threshold_selection_source = "validation"

    grid_cv = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=threshold_random_state)
        if isinstance(cv, int)
        else cv
    )

    if param_grid is not None:
        fitted_model = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=grid_cv,
            n_jobs=n_jobs,
            refit=True,
        )
    else:
        fitted_model = pipeline

    fitted_model.fit(X_fit, y_fit)

    if isinstance(fitted_model, GridSearchCV):
        print("Best parameters:", fitted_model.best_params_)
        print("Best cross-validation score:", fitted_model.best_score_)

    y_score, scores_are_probabilities = _positive_class_scores(fitted_model, X_test)

    if y_score is not None and threshold_mode in threshold_metric_map:
        y_threshold_score, threshold_scores_are_probabilities = _positive_class_scores(fitted_model, X_threshold)
        if y_threshold_score is not None:
            threshold_metric = threshold_metric_map[threshold_mode]
            selected_threshold, threshold_selection_scores = _best_threshold(
                y_threshold,
                y_threshold_score,
                threshold_values,
                threshold_metric,
            )
            print(f"Selected threshold by validation {threshold_metric}: {selected_threshold:.2f}")
            display(threshold_selection_scores)
        else:
            selected_threshold = 0.50
            threshold_selection_source = None
            print("Could not select a threshold because validation scores are unavailable.")
    elif threshold_mode is not None:
        selected_threshold = 0.50
        print("Could not select a threshold because fitted model scores are unavailable.")

    if y_score is not None:
        if isinstance(selected_threshold, str):
            raise RuntimeError("Threshold selection did not produce a numeric threshold.")
        y_pred = (y_score >= selected_threshold).astype(int)
    else:
        y_pred = fitted_model.predict(X_test)

    threshold_label = (
        f"{selected_threshold:.2f}"
        if isinstance(selected_threshold, (int, float, np.floating))
        else str(selected_threshold)
    )

    print(classification_report(y_test, y_pred, zero_division=0))
    if plot_confusion_matrix:
        cm = _plot_confusion_matrix(
            y_test,
            y_pred,
            f"Confusion Matrix for {model_name} at Threshold {threshold_label}",
            class_labels,
        )
    else:
        cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) else 0,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    }

    roc_auc = None
    threshold_points = None
    if y_score is not None:
        roc_auc = roc_auc_score(y_test, y_score)
        metrics["roc_auc"] = roc_auc
        metrics["average_precision"] = average_precision_score(y_test, y_score)
        print(f"Test ROC-AUC: {roc_auc:.3f}")
        print(f"Test average precision: {metrics['average_precision']:.3f}")

        if plot_roc_auc:
            roc_auc, threshold_points = _plot_roc_auc_with_thresholds(
                y_test,
                y_score,
                model_name,
                threshold_values=threshold_values,
            )
    elif plot_roc_auc:
        print("ROC-AUC chart skipped because the fitted model does not provide continuous scores.")

    return {
        "model_name": model_name,
        "estimator": fitted_model,
        "y_pred": y_pred,
        "y_score": y_score,
        "confusion_matrix": cm,
        "metrics": metrics,
        "roc_auc": roc_auc,
        "threshold_points": threshold_points,
        "selected_threshold": selected_threshold,
        "threshold_selection_scores": threshold_selection_scores,
        "threshold_selection_source": threshold_selection_source,
    }


__all__ = ["evaluate_model"]
