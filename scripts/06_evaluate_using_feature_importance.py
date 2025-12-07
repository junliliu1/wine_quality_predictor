"""
This script evaluates a trained Random Forest wine quality classifier
by analyzing feature importances.

It performs the following steps:
1. Loads processed wine data from a CSV file.
2. Loads a trained Random Forest model from disk.
3. Extracts feature importances from the model.
4. Creates and saves a horizontal bar plot of feature importances.
5. Saves a table of the top features (including cumulative importance).

Usage (from terminal):
$ python scripts/06_evaluate_using_feature_importance.py \
    --input-csv data/processed/wine_data_cleaned.csv \
    --model-path results/models/rf_wine_models.pkl \
    --output-dir results/evaluation

NOTE: The model should be trained on the same features as in the
processed CSV (e.g., via 04_train_wine_quality_classifier.py).
"""

from pathlib import Path
import pickle

import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    help="Directory to save feature importance figure and table.",
)
def main(input_csv: str, model_path: str, output_dir: str) -> None:
    """Evaluate feature importance for the trained Random Forest model."""
    print("=" * 60)
    print("STEP 6: EVALUATE MODEL (FEATURE IMPORTANCE)")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load processed data
    data = pd.read_csv(input_csv)
    print(f"Loaded {len(data)} samples from {input_csv}")

    # 2. Prepare feature matrix X (same as in training script)
    drop_cols = ["quality", "quality_category"]
    drop_cols = [c for c in drop_cols if c in data.columns]
    X = data.drop(columns=drop_cols)
    feature_names = X.columns.tolist()
    print(f"Using {len(feature_names)} features for importance analysis.")

    # 3. Load trained model
    with open(model_path, "rb") as f:
        rf_model = pickle.load(f)
    print(f"Loaded trained model from {model_path}")

    # Check that model has feature_importances_
    if not hasattr(rf_model, "feature_importances_"):
        raise AttributeError(
            "Loaded model does not have 'feature_importances_' attribute. "
            "Make sure it is a tree-based model such as RandomForestClassifier."
        )

    # 4. Build feature importance DataFrame
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    # 5. Plot feature importances (horizontal bar chart)
    plt.figure(figsize=(10, 8))

    # Normalize importances for coloring
    norm_importance = feature_importance["importance"] / feature_importance[
        "importance"
    ].max()
    colors = plt.cm.viridis(norm_importance)

    plt.barh(
        range(len(feature_importance)),
        feature_importance["importance"],
        color=colors,
    )
    plt.yticks(range(len(feature_importance)), feature_importance["feature"])
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title(
        "Random Forest Feature Importance Analysis",
        fontsize=14,
        fontweight="bold",
    )
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_path / "feature_importance_random_forest.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved feature importance figure to: {fig_path}")

    # 6. Save table of importances (including cumulative importance)
    feature_importance["cumulative_importance"] = feature_importance[
        "importance"
    ].cumsum()

    table_path = output_path / "feature_importance_table.csv"
    feature_importance.to_csv(table_path, index=False)
    print(f"Saved feature importance table to: {table_path}")

    # Print top 5 to console (similar to original notebook)
    top5 = feature_importance.head()
    print("\nTop 5 Most Important Features:")
    print(top5[["feature", "importance"]].to_string(index=False))
    print(
        f"\nCumulative importance of top 5 features: "
        f"{top5['importance'].sum():.3f}"
    )

    print("\nFeature importance evaluation complete!")


if __name__ == "__main__":
    main()
