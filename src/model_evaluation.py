"""
Model Evaluation Module

This module provides reusable functions for evaluating machine learning models,
including confusion matrix visualization, classification reports, and feature
importance analysis.

Functions:
    load_model: Load a trained model from pickle file
    load_splits: Load train/test splits from pickle file
    generate_confusion_matrix: Create and save confusion matrix visualization
    save_classification_report: Generate and save classification report
    evaluate_with_confusion_matrix: Complete confusion matrix evaluation workflow
    generate_feature_importance_plot: Create feature importance visualization
    save_feature_importance_table: Save feature importance as CSV
    evaluate_with_feature_importance: Complete feature importance evaluation workflow
"""

from pathlib import Path
from typing import Tuple, Any
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder


def load_model(model_path: str | Path) -> Any:
    """
    Load a trained model from pickle file.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained model (.pkl file)

    Returns
    -------
    Any
        Loaded model object

    Raises
    ------
    FileNotFoundError
        If model_path does not exist
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def load_splits(
    splits_path: str | Path
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load train/test splits from pickle file.

    Parameters
    ----------
    splits_path : str or Path
        Path to the splits pickle file

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]
        (X_train, X_test, y_train, y_test, label_encoder)

    Raises
    ------
    FileNotFoundError
        If splits_path does not exist
    """
    splits_path = Path(splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")

    with open(splits_path, "rb") as f:
        X_train, X_test, y_train, y_test, label_encoder = pickle.load(f)

    return X_train, X_test, y_train, y_test, label_encoder


def generate_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    output_path: str | Path,
    title: str = "Random Forest Confusion Matrix (Evaluation)",
    cmap: str = "Blues"
) -> None:
    """
    Generate and save confusion matrix visualization.

    Parameters
    ----------
    y_test : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : np.ndarray
        Array of class names
    output_path : str or Path
        Path to save the confusion matrix figure
    title : str, default="Random Forest Confusion Matrix (Evaluation)"
        Title for the plot
    cmap : str, default="Blues"
        Colormap for the heatmap

    Raises
    ------
    ValueError
        If y_test and y_pred have different lengths
    """
    if len(y_test) != len(y_pred):
        raise ValueError(f"y_test and y_pred must have same length: {len(y_test)} vs {len(y_pred)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix figure to: {output_path}")


def save_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    output_path: str | Path,
    title: str = "Random Forest Classification Report"
) -> str:
    """
    Generate and save classification report.

    Parameters
    ----------
    y_test : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    class_names : np.ndarray
        Array of class names
    output_path : str or Path
        Path to save the classification report
    title : str, default="Random Forest Classification Report"
        Title for the report

    Returns
    -------
    str
        The classification report text

    Raises
    ------
    ValueError
        If y_test and y_pred have different lengths
    """
    if len(y_test) != len(y_pred):
        raise ValueError(f"y_test and y_pred must have same length: {len(y_test)} vs {len(y_pred)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, target_names=class_names)

    with open(output_path, "w") as f:
        f.write(f"{title}\n\n")
        f.write(report)

    print(f"\nSaved classification report to: {output_path}")
    return report


def evaluate_with_confusion_matrix(
    model_path: str | Path,
    splits_path: str | Path,
    output_dir: str | Path
) -> None:
    """
    Complete confusion matrix evaluation workflow.

    This function:
    1. Loads train/test splits
    2. Loads trained model
    3. Generates predictions
    4. Creates confusion matrix visualization
    5. Generates classification report

    Parameters
    ----------
    model_path : str or Path
        Path to the trained model (.pkl)
    splits_path : str or Path
        Path to the splits pickle file
    output_dir : str or Path
        Directory to save evaluation outputs

    Raises
    ------
    FileNotFoundError
        If model_path or splits_path does not exist
    """
    print("=" * 60)
    print("STEP 5: EVALUATE MODEL")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load splits
    X_train, X_test, y_train, y_test, label_encoder = load_splits(splits_path)
    print(f"Loaded train/test splits from {splits_path}")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 2. Load model
    model = load_model(model_path)
    print(f"Loaded trained model from {model_path}")

    # 3. Predictions
    y_pred_test = model.predict(X_test)

    # 4. Confusion matrix
    cm_path = output_path / "confusion_matrix_random_forest.png"
    generate_confusion_matrix(
        y_test,
        y_pred_test,
        label_encoder.classes_,
        cm_path
    )

    # 5. Classification report
    report_path = output_path / "classification_report.txt"
    report = save_classification_report(
        y_test,
        y_pred_test,
        label_encoder.classes_,
        report_path
    )

    print("\nClassification Report:")
    print(report)
    print("\nModel evaluation complete!")


def generate_feature_importance_plot(
    feature_names: list,
    feature_importances: np.ndarray,
    output_path: str | Path,
    title: str = "Random Forest Feature Importance"
) -> pd.DataFrame:
    """
    Generate and save feature importance visualization.

    Parameters
    ----------
    feature_names : list
        List of feature names
    feature_importances : np.ndarray
        Array of feature importance values
    output_path : str or Path
        Path to save the figure
    title : str, default="Random Forest Feature Importance"
        Title for the plot

    Returns
    -------
    pd.DataFrame
        DataFrame with features and their importances, sorted by importance

    Raises
    ------
    ValueError
        If feature_names and feature_importances have different lengths
    """
    if len(feature_names) != len(feature_importances):
        raise ValueError(
            f"feature_names and feature_importances must have same length: "
            f"{len(feature_names)} vs {len(feature_importances)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame and sort
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importances,
    }).sort_values("importance", ascending=False)

    # Create plot
    plt.figure(figsize=(10, 8))

    norm = feature_importance["importance"] / feature_importance["importance"].max()
    colors = plt.cm.viridis(norm)

    plt.barh(
        feature_importance["feature"],
        feature_importance["importance"],
        color=colors,
    )
    plt.xlabel("Feature Importance", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved feature importance figure to: {output_path}")

    return feature_importance


def save_feature_importance_table(
    feature_importance: pd.DataFrame,
    output_path: str | Path
) -> None:
    """
    Save feature importance table with cumulative importance.

    Parameters
    ----------
    feature_importance : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    output_path : str or Path
        Path to save the CSV file

    Raises
    ------
    ValueError
        If required columns are missing
    """
    required_cols = ["feature", "importance"]
    missing_cols = [col for col in required_cols if col not in feature_importance.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add cumulative importance
    feature_importance["cumulative_importance"] = feature_importance["importance"].cumsum()

    feature_importance.to_csv(output_path, index=False)
    print(f"Saved feature importance table to: {output_path}")


def evaluate_with_feature_importance(
    input_csv: str | Path,
    model_path: str | Path,
    output_dir: str | Path
) -> None:
    """
    Complete feature importance evaluation workflow.

    This function:
    1. Loads processed dataset
    2. Loads trained model
    3. Extracts feature importances
    4. Generates feature importance plot
    5. Saves feature importance table
    6. Prints top features

    Parameters
    ----------
    input_csv : str or Path
        Path to the processed wine dataset (CSV)
    model_path : str or Path
        Path to the trained model (.pkl)
    output_dir : str or Path
        Directory to save plots and tables

    Raises
    ------
    FileNotFoundError
        If input_csv or model_path does not exist
    AttributeError
        If model doesn't have feature_importances_ attribute
    """
    print("=" * 60)
    print("STEP 6: MODEL EVALUATION — FEATURE IMPORTANCE")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    if not Path(input_csv).exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    data = pd.read_csv(input_csv)
    print(f"Loaded {len(data)} samples from: {input_csv}")

    # Feature selection
    drop_cols = [col for col in ["quality", "quality_category"] if col in data.columns]
    X = data.drop(columns=drop_cols)
    feature_names = X.columns.tolist()
    print(f"Using {len(feature_names)} predictor features for analysis.")

    # 2. Load model
    model = load_model(model_path)
    print(f"Loaded trained model from: {model_path}")

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            "Loaded model does not provide feature_importances_. "
            "Ensure you are loading a RandomForestClassifier or similar tree-based model."
        )

    # 3. Generate plot
    fig_path = output_path / "feature_importance_random_forest.png"
    feature_importance = generate_feature_importance_plot(
        feature_names,
        model.feature_importances_,
        fig_path
    )

    # 4. Save table
    table_path = output_path / "feature_importance_table.csv"
    save_feature_importance_table(feature_importance, table_path)

    # 5. Print top features
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
