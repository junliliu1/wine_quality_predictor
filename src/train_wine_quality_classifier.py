# src/train_wine_classifier.py
"""
This module contains reusable functions for training a Random Forest classifier
on the cleaned wine quality dataset. 

Functions included:
- load_data: Load features and target from a CSV file
- split_data: Split dataset into train/test sets and encode labels
- train_model: Train a Random Forest classifier
- evaluate_model: Evaluate model performance (accuracy, OOB, CV scores)
- save_model: Save trained model to disk
- save_splits: Save train/test splits to disk
- ensure_dir_exists: Create directory if it does not exist

"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def load_data(input_csv: str):
    data = pd.read_csv(input_csv)
    X = data.drop(columns=["quality", "quality_category"])
    y = data["quality_category"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    return X_train, X_test, y_train, y_test, le

def train_model(X_train, y_train, random_state=42):
    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=random_state, oob_score=True, n_jobs=1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(rf_model, X_train, X_test, y_train, y_test):
    """Return metrics as a dict instead of printing."""
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    oob_score = getattr(rf_model, "oob_score_", None)  # Safe for dummy models
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=4, scoring="accuracy")
    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "oob_score": oob_score,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    }

def save_model(model, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

def save_splits(splits_tuple, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(splits_tuple, f)

def ensure_dir_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)
