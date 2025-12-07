"""
This script performs hyperparameter tuning for a Random Forest classifier
to predict wine quality categories from physicochemical properties.

It performs the following steps:
1. Loads processed wine data from a CSV file.
2. Splits the data into features and encoded quality_category labels.
3. Splits data into training and test sets (with stratification).
4. Runs GridSearchCV to tune Random Forest hyperparameters.
5. Reports the best hyperparameters and cross-validation score.
6. Evaluates the optimized model on the test set:
   - Test accuracy, precision, recall, F1-score
   - Confusion matrix (saved as a PNG figure)
7. Saves the optimized model to a specified file path.

Usage (from terminal):
$ python scripts/08_tune_random_forest_hyperparameters.py \
    --input-csv data/processed/wine_data_cleaned.csv \
    --output-model results/models/rf_wine_model_optimized.pkl \
    --output-dir results/evaluation \
    --test-size 0.2 \
    --random-state 42
"""

from pathlib import Path
import pickle

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the processed wine data CSV.",
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
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing (must match other scripts).",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
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
    help="Number of parallel jobs for GridSearchCV and RandomForest (default: -1 = all cores).",
)
def main(
    input_csv: str,
    output_model: str,
    output_dir: str,
    test_size: float,
    random_state: int,
    cv_folds: int,
    n_jobs: int,
) -> None:
    """Tune Random Forest hyperparameters and evaluate optimized model."""
    print("=" * 60)
    print("STEP 8: HYPERPARAMETER TUNING FOR RANDOM FOREST")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load processed data
    data = pd.read_csv(input_csv)
    print(f"Loaded {len(data)} samples from {input_csv}")

    if "quality_category" not in data.columns:
        raise ValueError(
            "Column 'quality_category' not found in data. "
            "Make sure you ran the cleaning script that creates this column."
        )

    # 2. Prepare features X and encoded target y (quality_category)
    drop_cols = ["quality", "quality_category"]
    drop_cols = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=drop_cols)
    y = data["quality_category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    print(f"\nSplit data: {len(X_train)} training samples, {len(X_test)} test samples")
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {counts[i]} ({counts[i] / len(y_train) * 100:.1f}%)")

    print("\nClass distribution in test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {counts[i]} ({counts[i] / len(y_test) * 100:.1f}%)")

    # 4. Define parameter grid for Random Forest
    print("\nStarting Random Forest hyperparameter tuning...")
    print("This may take several minutes depending on data size and parameter grid.\n")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    # 5. Create GridSearchCV object
    base_rf = RandomForestClassifier(
        random_state=random_state, oob_score=True, n_jobs=n_jobs
    )

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    rf_grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=1,
    )

    # 6. Fit GridSearchCV
    rf_grid.fit(X_train, y_train)

    print("\nBest parameters found:")
    for param, value in rf_grid.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\nBest cross-validation score: {rf_grid.best_score_:.4f}")

    # Save tuning summary to a text file
    tuning_summary_path = output_path / "rf_hyperparameter_tuning_results.txt"
    with open(tuning_summary_path, "w") as f:
        f.write("Random Forest Hyperparameter Tuning Results\n\n")
        f.write("Best parameters:\n")
        for param, value in rf_grid.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nBest cross-validation accuracy: {rf_grid.best_score_:.4f}\n")
    print(f"\nSaved tuning summary to: {tuning_summary_path}")

    # 7. Evaluate optimized Random Forest on test set
    rf_optimized = rf_grid.best_estimator_
    y_pred_optimized = rf_optimized.predict(X_test)

    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_optimized, average="weighted"
    )

    print("\nOptimized Random Forest Performance on Test Set:")
    print(f"  Test Accuracy: {accuracy_optimized:.4f}")
    print(f"  Precision:     {precision:.4f}")
    print(f"  Recall:        {recall:.4f}")
    print(f"  F1-Score:      {f1:.4f}")

    # 8. Confusion matrix for optimized model
    cm_optimized = confusion_matrix(y_test, y_pred_optimized)
    class_names = label_encoder.classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_optimized,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(
        "Random Forest Confusion Matrix (Optimized Model)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = output_path / "confusion_matrix_random_forest_optimized.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix for optimized model to: {cm_path}")

    # 9. Save optimized model
    output_model_path = Path(output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_model_path, "wb") as f:
        pickle.dump(rf_optimized, f)
    print(f"\nSaved optimized model to: {output_model_path}")

    print("\nHyperparameter tuning and optimized model evaluation complete!")


if __name__ == "__main__":
    main()
