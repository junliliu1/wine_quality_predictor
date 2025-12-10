"""
Hyperparameter tuning for Random Forest using pre-split data.

This script performs the following steps:
1. Loads train/test splits and label encoder from splits.pkl.
2. Runs GridSearchCV to tune Random Forest hyperparameters.
3. Reports the best hyperparameters and cross-validation score.
4. Evaluates the optimized model on the test set:
   - Test accuracy, precision, recall, F1-score
   - Confusion matrix (saved as PNG)
5. Saves the optimized model and test metrics to files.

Usage:
$ python scripts/07_tune_random_forest_hyperparameters.py \
    --splits-pkl results/splits.pkl \
    --output-model results/models/rf_wine_model_optimized.pkl \
    --output-dir results/evaluation \
    --cv-folds 5 \
    --n-jobs -1
"""

from pathlib import Path
import pickle
import click
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


@click.command()
@click.option(
    "--splits-pkl",
    type=click.Path(exists=True),
    required=True,
    help="Path to pickle file containing X_train, X_test, y_train, y_test, label_encoder",
)
@click.option(
    "--output-model",
    type=click.Path(),
    default="results/models/rf_wine_model_optimized.pkl",
    help="Path to save the optimized Random Forest model (.pkl).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/evaluation",
    help="Directory to save evaluation figures and tuning results.",
)
@click.option(
    "--cv-folds",
    type=int,
    default=5,
    help="Number of cross-validation folds for GridSearchCV.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=-1,
    help="Number of parallel jobs for GridSearchCV and RandomForest (-1 = all cores).",
)
def main(splits_pkl, output_model, output_dir, cv_folds, n_jobs):
    print("=" * 60)
    print("STEP 8: HYPERPARAMETER TUNING FOR RANDOM FOREST")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-split data
    with open(splits_pkl, "rb") as f:
        X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)

    print(f"Loaded pre-split data from {splits_pkl}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}\n")

    # Show class distribution
    print("Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {counts[i]} ({counts[i]/len(y_train)*100:.1f}%)")

    print("\nClass distribution in test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {counts[i]} ({counts[i]/len(y_test)*100:.1f}%)")

    # 2. Define parameter grid
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    # 3. Create Random Forest with class_weight="balanced"
    base_rf = RandomForestClassifier(
        random_state=42,
        oob_score=True,
        n_jobs=n_jobs,
        class_weight="balanced"  # <-- helps with minority classes
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    rf_grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=1,
    )

    # 4. Fit GridSearchCV
    rf_grid.fit(X_train, y_train)

    print("\nBest parameters found:")
    for param, value in rf_grid.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest cross-validation score: {rf_grid.best_score_:.4f}")

    # Save tuning summary
    tuning_summary_path = output_path / "rf_hyperparameter_tuning_results.txt"
    with open(tuning_summary_path, "w") as f:
        f.write("Random Forest Hyperparameter Tuning Results\n\n")
        f.write("Best parameters:\n")
        for param, value in rf_grid.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nBest cross-validation accuracy: {rf_grid.best_score_:.4f}\n")
    print(f"\nSaved tuning summary to: {tuning_summary_path}")

    # 5. Evaluate optimized model
    rf_optimized = rf_grid.best_estimator_
    y_pred = rf_optimized.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print("\nOptimized Random Forest Performance on Test Set:")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1-Score:      {f1:.4f}")

    # Save test metrics to JSON
    test_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    metrics_path = output_path / "rf_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"\nSaved test set metrics to: {metrics_path}")

    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title("Random Forest Confusion Matrix (Optimized Model)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = output_path / "confusion_matrix_random_forest_optimized.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix to: {cm_path}")

    # 7. Save optimized model
    output_model_path = Path(output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model_path, "wb") as f:
        pickle.dump(rf_optimized, f)
    print(f"\nSaved optimized model to: {output_model_path}")

    print("\nHyperparameter tuning and evaluation complete!")


if __name__ == "__main__":
    main()
