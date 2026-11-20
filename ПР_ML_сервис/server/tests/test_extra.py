"""Повышенная сложность: pytest -m extra."""

import pandas as pd
import pytest
from conftest import DATA_DIR, EXAMPLES, VERSION_1, VERSION_2, check_model_loaded, expected, sample, seed_prediction

pytestmark = pytest.mark.extra


def test_batch_csv(client, bundle):
    path = DATA_DIR / "batch.csv"
    response = client.post("/api/predict/batch", files={"file": ("batch.csv", path.read_bytes(), "text/csv")})
    check_model_loaded(response)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"] == 6
    assert [item["line"] for item in body["predictions"]] == [2, 3, 4, 6, 7]
    assert [(e["line"], e["field"]) for e in body["errors"]] == [(5, "road_temp")]
    frame = pd.read_csv(path).drop(index=3)
    want = [{"сухой асфальт": "dry", "мокрый асфальт": "wet", "грунт": "dirt", "снег и лед": "snow"}[c]
            for c in bundle["pipeline"].predict(frame[bundle["features"]])]
    assert [item["surface"] for item in body["predictions"]] == want
    assert sum(body["counts"].values()) == 5


def test_search_by_feature(client, db):
    for air in (-5, 0, 10):
        seed_prediction(db, "dry", features=sample(air_temp=air))
    search = "/api/predictions/search"
    assert [r["id"] for r in client.get(search, params={"feature": "air_temp", "max": 0}).json()] == [1, 2]
    assert [r["id"] for r in client.get(search, params={"feature": "air_temp", "min": 0, "max": 5}).json()] == [2]
    assert client.get(search, params={"feature": "rec_speed", "max": 0}).status_code == 422


def test_compare_versions(client):
    example = EXAMPLES[3]          # мокрый асфальт: первая версия права, вторая перестраховывается
    response = client.post("/api/predict/compare", json=example["features"])
    check_model_loaded(response)
    assert response.status_code == 200, response.text
    answers = response.json()
    assert [a["model_version"] for a in answers] == [VERSION_1, VERSION_2]
    assert [a["surface"] for a in answers] == [expected(example, VERSION_1)[0], expected(example, VERSION_2)[0]]
    assert answers[1]["probabilities"]["snow"] > answers[0]["probabilities"]["snow"]
