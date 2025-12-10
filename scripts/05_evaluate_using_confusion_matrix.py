"""
Evaluate a trained Random Forest wine quality classifier using
a confusion matrix and classification report.

This script performs the following steps:
1. Loads train/test splits from pickle file.
2. Loads a trained Random Forest model from disk.
3. Generates predictions on the test set.
4. Computes and saves:
   - Confusion matrix (as a PNG figure)
   - Classification report (as a text file)

Usage (from terminal):
$ python scripts/05_evaluate_using_confusion_matrix.py \
    --model-path results/models/rf_wine_models.pkl \
    --splits-path results/splits.pkl \
    --output-dir results/evaluation
"""

from pathlib import Path
import pickle

import click
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

@click.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the trained Random Forest model (.pkl)",
)
@click.option(
    "--splits-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the pickle file containing train/test splits",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="results/evaluation",
    help="Directory to save evaluation figures and reports",
)
def main(model_path: str, splits_path: str, output_dir: str) -> None:
    """Evaluate the Random Forest classifier on the test set."""
    print("=" * 60)
    print("STEP 5: EVALUATE MODEL")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load train/test splits
    with open(splits_path, "rb") as f:
        X_train, X_test, y_train, y_test, classes = pickle.load(f)
    print(f"Loaded train/test splits from {splits_path}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 2. Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded trained model from {model_path}")

    # 3. Predictions on test set
    y_pred_test = model.predict(X_test)

    # 4. Confusion matrix
    #cm = confusion_matrix(y_test, y_pred_test, labels=classes)

    # Extract integers + class names
    labels = list(range(len(classes.classes_)))  # e.g., [0,1,2]
    names = classes.classes_                    # e.g., ["Low","Medium","High"]

    #labels = classes.classes_   # extract actual class names
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
    )
    plt.title("Random Forest Confusion Matrix (Evaluation)", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    cm_path = output_path / "confusion_matrix_random_forest.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix figure to: {cm_path}")

    # 5. Classification report
    report = classification_report(y_test, y_pred_test, target_names=names)
    print("\nClassification Report:")
    print(report)

    report_path = output_path / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Random Forest Classification Report\n\n")
        f.write(report)
    print(f"\nSaved classification report to: {report_path}")

    print("\nModel evaluation complete!")


if __name__ == "__main__":
    main()
