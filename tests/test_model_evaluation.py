import os
os.environ["MPLBACKEND"] = "Agg" 

import pickle
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

from src.model_evaluation import (
    load_model,
    load_splits,
    generate_confusion_matrix,
    save_classification_report,
    evaluate_with_confusion_matrix,
    generate_feature_importance_plot,
    save_feature_importance_table,
    evaluate_with_feature_importance,
)


class FakePredictModel:
    def __init__(self, y_pred):
        self._y_pred = list(y_pred)

    def predict(self, X):
        return np.array(self._y_pred[: len(X)])


class FakeFeatureImportanceModel:
    def __init__(self, importances):
        self.feature_importances_ = np.array(importances, dtype=float)


def test_load_model_success(tmp_path):
    obj = {"a": 1, "b": "ok"}
    p = tmp_path / "model.pkl"
    with open(p, "wb") as f:
        pickle.dump(obj, f)

    loaded = load_model(p)
    assert loaded == obj


def test_load_model_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pkl")


def test_load_splits_success(tmp_path):
    X_train = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    X_test = pd.DataFrame({"f1": [5], "f2": [6]})
    y_train = np.array([0, 1])
    y_test = np.array([1])

    le = LabelEncoder()
    le.fit(["Low", "High"])

    p = tmp_path / "splits.pkl"
    with open(p, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test, le), f)

    Xtr, Xte, ytr, yte, le2 = load_splits(p)
    assert Xtr.shape == (2, 2)
    assert Xte.shape == (1, 2)
    assert ytr.tolist() == [0, 1]
    assert yte.tolist() == [1]
    assert le2.classes_.tolist() == ["High", "Low"] or le2.classes_.tolist() == ["Low", "High"]


def test_load_splits_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path / "missing.pkl")


def test_generate_confusion_matrix_creates_file(tmp_path):
    y_test = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    class_names = np.array(["A", "B"])

    out = tmp_path / "cm.png"
    generate_confusion_matrix(y_test, y_pred, class_names, out)
    assert out.exists()


def test_generate_confusion_matrix_length_mismatch_raises(tmp_path):
    y_test = np.array([0, 1, 1])
    y_pred = np.array([0, 1])
    class_names = np.array(["A", "B"])

    with pytest.raises(ValueError):
        generate_confusion_matrix(y_test, y_pred, class_names, tmp_path / "cm.png")


def test_save_classification_report_creates_file_and_returns_text(tmp_path):
    y_test = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    class_names = np.array(["A", "B"])

    out = tmp_path / "report.txt"
    report_text = save_classification_report(y_test, y_pred, class_names, out, title="My Report")
    assert out.exists()
    saved = out.read_text()
    assert "My Report" in saved
    assert isinstance(report_text, str)
    assert "precision" in report_text or "f1-score" in report_text


def test_save_classification_report_length_mismatch_raises(tmp_path):
    y_test = np.array([0, 1, 1])
    y_pred = np.array([0, 1])
    class_names = np.array(["A", "B"])

    with pytest.raises(ValueError):
        save_classification_report(y_test, y_pred, class_names, tmp_path / "r.txt")


def test_evaluate_with_confusion_matrix_workflow_outputs_files(tmp_path):
    X_train = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 1, 0, 0]})
    X_test = pd.DataFrame({"f1": [4, 5], "f2": [1, 0]})
    y_train = np.array([0, 1, 0, 1])
    y_test = np.array([0, 1])

    le = LabelEncoder()
    le.fit(["Low", "High"])

    splits_p = tmp_path / "splits.pkl"
    with open(splits_p, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test, le), f)

    model = FakePredictModel(y_test)

    model_p = tmp_path / "model.pkl"
    with open(model_p, "wb") as f:
        pickle.dump(model, f)

    out_dir = tmp_path / "eval"
    evaluate_with_confusion_matrix(model_p, splits_p, out_dir)

    assert (out_dir / "confusion_matrix_random_forest.png").exists()
    assert (out_dir / "classification_report.txt").exists()


def test_generate_feature_importance_plot_creates_file_and_returns_sorted_df(tmp_path):
    feature_names = ["f1", "f2", "f3"]
    importances = np.array([0.2, 0.5, 0.3])

    out = tmp_path / "fi.png"
    df = generate_feature_importance_plot(feature_names, importances, out)
    assert out.exists()
    assert list(df.columns) == ["feature", "importance"]
    assert df.iloc[0]["importance"] == pytest.approx(0.5)


def test_generate_feature_importance_plot_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_feature_importance_plot(["f1", "f2"], np.array([0.1]), tmp_path / "fi.png")


def test_save_feature_importance_table_creates_csv_and_has_cumulative(tmp_path):
    df = pd.DataFrame({"feature": ["a", "b"], "importance": [0.7, 0.3]})
    out = tmp_path / "fi.csv"
    save_feature_importance_table(df, out)

    assert out.exists()
    saved = pd.read_csv(out)
    assert "cumulative_importance" in saved.columns
    assert saved["cumulative_importance"].iloc[-1] == pytest.approx(1.0)


def test_save_feature_importance_table_missing_cols_raises(tmp_path):
    df = pd.DataFrame({"feature": ["a", "b"], "wrong": [1, 2]})
    with pytest.raises(ValueError):
        save_feature_importance_table(df, tmp_path / "fi.csv")


def test_evaluate_with_feature_importance_workflow_outputs_files(tmp_path):
    data = pd.DataFrame(
        {
            "quality": [5, 6, 7],
            "quality_category": ["Low", "Medium", "High"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [0.1, 0.2, 0.3],
            "f3": [10, 11, 12],
        }
    )
    csv_p = tmp_path / "wine.csv"
    data.to_csv(csv_p, index=False)

    model = FakeFeatureImportanceModel([0.2, 0.5, 0.3])
    model_p = tmp_path / "model.pkl"
    with open(model_p, "wb") as f:
        pickle.dump(model, f)

    out_dir = tmp_path / "eval2"
    evaluate_with_feature_importance(csv_p, model_p, out_dir)

    assert (out_dir / "feature_importance_random_forest.png").exists()
    assert (out_dir / "feature_importance_table.csv").exists()