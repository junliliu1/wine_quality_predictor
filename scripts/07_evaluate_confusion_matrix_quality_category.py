"""
Evaluate a trained Random Forest wine quality classifier using a
confusion matrix and classification report.

This script performs the following steps:
1. Loads processed wine data from a CSV file.
2. Encodes the quality_category labels using LabelEncoder.
3. Splits the data into training and test sets (same as in training).
4. Loads a trained Random Forest model from disk.
5. Generates predictions on the test set.
6. Computes and saves:
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
(04_train_wine_quality_classifier.py) so the train/test split matches.
"""

from pathlib import Path
import pickle

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.option(
    "--input-csv",
    type=click.Path(exists=True),
    required=True,
    help="Path to the processed wine data CSV.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained Random Forest model (.pkl).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/evaluation",
    help="Directory to save evaluation figures and reports.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.2,
    help="Proportion of data to use for testing (must match training).",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility (must match training).",
)
def main(
    input_csv: str,
    model_path: str,
    output_dir: str,
    test_size: float,
    random_state: int,
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

    if "quality_category" not in data.columns:
        raise ValueError(
            "Column 'quality_category' not found in data. "
            "Make sure you ran the cleaning script that creates this column."
        )

    # 2. Prepare features X and encoded target y
    drop_cols = ["quality", "quality_category"]
    drop_cols = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=drop_cols)
    y = data["quality_category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Train-test split with stratification (must match training)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    print(f"Split data: {len(X_train)} training samples, {len(X_test)} test samples")

    print("\nClass distribution in test set:")
    unique, counts = np.unique(y_test, return_counts=True)
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label}: {counts[i]} ({counts[i] / len(y_test) * 100:.1f}%)")

    # 4. Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"\nLoaded trained model from {model_path}")

    # 5. Predictions on test set
    y_pred_test = model.predict(X_test)

    # 6. Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    class_names = label_encoder.classes_

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(
        "Random Forest Confusion Matrix (Initial Model)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = output_path / "confusion_matrix_random_forest_initial.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix figure to: {cm_path}")

    # 7. Classification report
    report = classification_report(
        y_test, y_pred_test, target_names=class_names
    )
    print("\nClassification Report:")
    print(report)

    report_path = output_path / "classification_report_random_forest_initial.txt"
    with open(report_path, "w") as f:
        f.write("Random Forest Classification Report (Initial Model)\n\n")
        f.write(report)

    print(f"\nSaved classification report to: {report_path}")
    print("\nModel evaluation (confusion matrix + classification report) complete!")


if __name__ == "__main__":
    main()
