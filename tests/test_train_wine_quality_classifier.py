import pandas as pd
import pytest
import tempfile
from pathlib import Path
from src.train_wine_quality_classifier import load_data, split_data, train_model, save_model, save_splits, evaluate_model

def test_load_data(tmp_path):
    file = tmp_path / "test.csv"
    file.write_text("quality,quality_category,feature1,feature2\n5,Low (3-5),1,2\n6,Medium (6-7),3,4")
    X, y = load_data(file)
    assert X.shape == (2, 2)
    assert y.tolist() == ["Low (3-5)", "Medium (6-7)"]

def test_split_data():
    import numpy as np
    X = pd.DataFrame({"f1":[1,2,3,4,5,6], "f2":[7,8,9,10,11,12]})
    y = pd.Series(["Low (3-5)", "Low (3-5)", "Medium (6-7)", "Medium (6-7)", "High (8-9)", "High (8-9)"])
    X_train, X_test, y_train, y_test, le = split_data(X, y, test_size=0.5, random_state=42)
    assert len(X_train) == 3 and len(X_test) == 3

def test_train_model():
    import pandas as pd
    X_train = pd.DataFrame({"f1":[1,2], "f2":[3,4]})
    y_train = [0,1]
    model = train_model(X_train, y_train)
    assert hasattr(model, "predict")

def test_save_model(tmp_path):
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier()
    model.fit([[0],[1]],[0,1])
    path = tmp_path / "model.pkl"
    save_model(model, path)
    assert path.exists()

def test_save_splits(tmp_path):
    splits = ([1,2],[3,4],[0,1],[1,0], None)
    path = tmp_path / "splits.pkl"
    save_splits(splits, path)
    assert path.exists()

def test_evaluate_model():
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    X_train = pd.DataFrame({"f1": range(16)})
    X_test = pd.DataFrame({"f1": range(16)})
    y_train = [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
    y_test = y_train
    model = DummyClassifier()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    assert "train_acc" in metrics
    assert "test_acc" in metrics
    assert "cv_mean" in metrics
