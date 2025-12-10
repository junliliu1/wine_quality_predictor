"""
Evaluate a trained Random Forest wine quality classifier
by analyzing feature importances.

This script performs the following steps:
1. Loads the cleaned, processed wine dataset.
2. Loads a trained Random Forest model from disk.
3. Extracts and ranks feature importances.
4. Generates a horizontal bar plot of feature importance.
5. Saves both a figure and a CSV table including cumulative importance.

Usage:
    python scripts/06_evaluate_using_feature_importance.py \
        --input-csv data/processed/wine_data_cleaned.csv \
        --model-path results/models/rf_wine_models.pkl \
        --output-dir results/evaluation

NOTE:
    The model must be trained on the same feature set as the input CSV.
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
    help="Path to the processed wine dataset (CSV).",
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
    help="Directory to save plots and tables.",
)
def main(input_csv: str, model_path: str, output_dir: str) -> None:
    """Analyze and visualize feature importance for a trained Random Forest model."""
    print("=" * 60)
    print("STEP 6: MODEL EVALUATION — FEATURE IMPORTANCE")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. Load processed dataset
    # ----------------------------------------------------
    data = pd.read_csv(input_csv)
    print(f"Loaded {len(data)} samples from: {input_csv}")

    # Feature selection (must match training script)
    drop_cols = [col for col in ["quality", "quality_category"] if col in data.columns]
    X = data.drop(columns=drop_cols)
    feature_names = X.columns.tolist()

    print(f"Using {len(feature_names)} predictor features for analysis.")

    # ----------------------------------------------------
    # 2. Load trained model
    # ----------------------------------------------------
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded trained model from: {model_path}")

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "Loaded model does not provide feature_importances_. "
            "Ensure you are loading a RandomForestClassifier or similar tree-based model."
        )

    # ----------------------------------------------------
    # 3. Compute feature importance DataFrame
    # ----------------------------------------------------
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # ----------------------------------------------------
    # 4. Create horizontal bar chart
    # ----------------------------------------------------
    plt.figure(figsize=(10, 8))

    norm = feature_importance["importance"] / feature_importance["importance"].max()
    colors = plt.cm.viridis(norm)

    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color=colors,
    )
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fig_path = output_path / "feature_importance_random_forest.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved feature importance figure to: {fig_path}")

    # ----------------------------------------------------
    # 5. Save feature importance table
    # ----------------------------------------------------
    feature_importance["cumulative_importance"] = feature_importance["importance"].cumsum()

    table_path = output_path / "feature_importance_table.csv"
    feature_importance.to_csv(table_path, index=False)
    print(f"Saved feature importance table to: {table_path}")

    # ----------------------------------------------------
    # 6. Print top features
    # ----------------------------------------------------
    print("\nTop 5 Most Important Features:")
    print(
        feature_importance[["feature", "importance"]]
        .head(5)
        .to_string(index=False)
    )

    print(
        f"\nCumulative importance of top 5 features: "
        f"{feature_importance['importance'].head(5).sum():.3f}"
    )

    print("\nFeature importance evaluation complete!")


if __name__ == "__main__":
    main()
