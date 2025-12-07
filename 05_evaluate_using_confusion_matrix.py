"""
Evaluate a trained Random Forest wine quality classifier using a
confusion matrix and classification report.

This script performs the following steps:
1. Loads processed wine data from a CSV file.
2. Loads a trained Random Forest model from disk.
3. Splits the data into training and test sets (same as in training).
4. Generates predictions on the test set.
5. Computes and saves:
   - Confusion matrix (as a PNG figure)
   - Classification report (as a text file)

Usage (from terminal):
$ python scripts/05_evaluate_using_confusion_matrix.py \
    --input-csv data/processed/wine_data_cleaned.csv \
    --model-path results/models/rf_wine_models.pkl \
    --output-dir results/evaluation \
    --test-size 0.2 \
    --random-state 42

NOTE: Use the SAME test_size and random_state as in the training script
to ensure the train/test split is identical.
"""

from pathlib import Path
import pickle

import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the processed wine data CSV",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained Random Forest model (.pkl)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/evaluation",
    help="Directory to save evaluation figures and reports",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing (must match training)",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility (must match training)",
)
@click.option(
    "--target-column",
    type=str,
    default="quality",
    help="Name of the target column in the data "
         "(e.g., 'quality' or 'quality_category').",
)
def main(
    input_csv: str,
    model_path: str,
    output_dir: str,
    test_size: float,
    random_state: int,
    target_column: str,
) -> None:
    """Evaluate the Random Forest classifier using a confusion matrix."""
    print("=" * 60)
    print("STEP 5: EVALUATE MODEL (CONFUSION MATRIX)")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load processed data
    data = pd.read_csv(input_csv)
    print(f"Loaded {len(data)} samples from {input_csv}")

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data. "
            f"Available columns: {list(data.columns)}"
        )

    # 2. Split into features (X) and target (y)
    drop_cols = [target_column]
    # Avoid leaking quality_category as a feature if present
    if "quality_category" in data.columns and "quality_category" not in drop_cols:
        drop_cols.append("quality_category")

    X = data.drop(columns=drop_cols)
    y = data[target_column]

    # 3. Train-test split (must match settings used in training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

    # 4. Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded trained model from {model_path}")

    # 5. Predictions on test set
    y_pred_test = model.predict(X_test)

    # 6. Confusion matrix
    class_names = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred_test, labels=class_names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Random Forest Confusion Matrix (Evaluation)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = output_path / "confusion_matrix_random_forest.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to: {cm_path}")

    # 7. Classification report
    report = classification_report(
        y_test, y_pred_test
    )
    print("\nClassification Report:")
    print(report)

    report_path = output_path / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Random Forest Classification Report\n\n")
        f.write(report)
    print(f"\nSaved classification report to: {report_path}")

    print("\nModel evaluation (confusion matrix + classification report) complete!")


if __name__ == "__main__":
    main()
