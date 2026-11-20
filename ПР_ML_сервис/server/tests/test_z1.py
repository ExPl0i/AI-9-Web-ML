"""З1. Загрузка модели один раз при старте: load_model в app/ml.py и lifespan в app/main.py."""

import pathlib

import joblib
import pytest
import sklearn
from conftest import VERSION_1
from fastapi.testclient import TestClient

import app.main as main
from app.config import settings
from app.ml import load_model


def test_load_model_returns_bundle():
    bundle = load_model(settings.model_path)
    assert isinstance(bundle, dict)
    assert bundle["version"] == VERSION_1
    assert len(bundle["features"]) == 14
    assert hasattr(bundle["pipeline"], "predict_proba")


def test_load_model_accepts_string_path():
    assert load_model(str(settings.model_path))["version"] == VERSION_1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        load_model(tmp_path / "nope.joblib")
    assert "nope.joblib" in str(info.value), "в тексте ошибки должен быть путь к файлу"


def test_wrong_sklearn_version(tmp_path, bundle):
    path = tmp_path / "old.joblib"
    joblib.dump({**bundle, "sklearn_version": "0.24.2"}, path)
    with pytest.raises(RuntimeError) as info:
        load_model(path)
    text = str(info.value)
    assert "0.24.2" in text and sklearn.__version__ in text, "в тексте ошибки должны быть обе версии"


def test_missing_keys(tmp_path, bundle):
    path = tmp_path / "broken.joblib"
    joblib.dump({"pipeline": bundle["pipeline"]}, path)
    with pytest.raises(RuntimeError) as info:
        load_model(path)
    assert "version" in str(info.value)


def test_loaded_once_at_startup(monkeypatch):
    calls = []

    def spy(path):
        calls.append(pathlib.Path(path))
        return load_model(path)

    monkeypatch.setattr(main, "load_model", spy)
    with TestClient(main.app) as client:
        assert calls.count(settings.model_path) == 1, "модель должна загрузиться при старте сервера, до запросов"
        for _ in range(5):
            response = client.get("/api/model")
            assert response.status_code == 200, response.text
        assert calls.count(settings.model_path) == 1, "модель загружается заново на запрос"


def test_model_info():
    with TestClient(main.app) as client:
        assert main.app.state.model["version"] == VERSION_1, "модель должна лежать в app.state.model"
        info = client.get("/api/model").json()
    assert info["version"] == VERSION_1
    assert len(info["features"]) == 14
    assert set(info["classes"]) == {"dry", "wet", "dirt", "snow"}
    assert info["sklearn_version"] == sklearn.__version__
