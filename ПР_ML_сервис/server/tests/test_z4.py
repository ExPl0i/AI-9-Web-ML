"""З4. Прогноз: predict_one в app/ml.py и POST /api/predict с моделью ответа PredictionOut."""

import random

import pytest
from conftest import EXAMPLES, NAMES, ORDER, VERSION_1, check_model_loaded, close_probabilities, expected

from app.ml import predict_one

API = "/api/predict"


@pytest.mark.parametrize("number", range(len(EXAMPLES)))
def test_predict_one_matches_reference(bundle, number):
    example = EXAMPLES[number]
    surface, probabilities = predict_one(bundle, example["features"])
    want_surface, want_probabilities = expected(example)
    assert surface == want_surface
    assert close_probabilities(probabilities, want_probabilities)


def test_probabilities_format(bundle):
    _, probabilities = predict_one(bundle, EXAMPLES[3]["features"])
    assert list(probabilities) == ORDER, "порядок вероятностей: dry, wet, dirt, snow"
    assert abs(sum(probabilities.values()) - 1) < 1e-3
    assert all(round(p, 4) == p for p in probabilities.values()), "вероятности округлены до 4 знаков"


def test_feature_order_does_not_matter(bundle):
    features = list(EXAMPLES[4]["features"].items())
    random.Random(1).shuffle(features)
    assert predict_one(bundle, dict(features)) == predict_one(bundle, EXAMPLES[4]["features"])


def test_predict_route(client):
    for example in EXAMPLES:
        response = client.post(API, json=example["features"])
        check_model_loaded(response)
        assert response.status_code == 200, response.text
        body = response.json()
        want_surface, want_probabilities = expected(example)
        assert body["surface"] == want_surface
        assert body["surface_name"] == NAMES[want_surface]
        assert close_probabilities(body["probabilities"], want_probabilities)
        assert body["model_version"] == VERSION_1
        assert 0 <= body["elapsed_ms"] < 1000


def test_response_has_no_input(client):
    response = client.post(API, json=EXAMPLES[0]["features"])
    check_model_loaded(response)
    assert "speed" not in response.json() and "features" not in response.json()
