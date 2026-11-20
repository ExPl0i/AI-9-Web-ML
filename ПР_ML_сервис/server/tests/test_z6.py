"""З6. Обратная связь от водителя и доля верных прогнозов."""

import psycopg
from conftest import SAMPLE, check_model_loaded, seed_prediction

ACCURACY = "/api/predictions/accuracy"


def feedback(client, prediction_id, actual):
    return client.post(f"/api/predictions/{prediction_id}/feedback", json={"actual": actual})


def test_feedback_correct(client, db):
    response = client.post("/api/predict", json=SAMPLE)
    check_model_loaded(response)
    predicted = response.json()["surface"]
    response = feedback(client, 1, predicted)
    assert response.status_code == 200, response.text
    assert response.json() == {"id": 1, "surface": predicted, "actual": predicted, "correct": True}
    row = db.execute("SELECT actual, feedback_at FROM predictions WHERE id = 1").fetchone()
    assert row["actual"] == predicted and row["feedback_at"] is not None


def test_feedback_wrong_and_repeat(client, db):
    seed_prediction(db, "dry")
    assert feedback(client, 1, "snow").json()["correct"] is False
    assert feedback(client, 1, "dry").json()["correct"] is True, "повторная отметка заменяет прежнюю"
    assert db.execute("SELECT actual FROM predictions WHERE id = 1").fetchone()["actual"] == "dry"


def test_feedback_not_found(client, db):
    response = feedback(client, 999, "dry")
    assert response.status_code == 404
    assert "999" in response.json()["error"]


def test_feedback_validation(client, db):
    seed_prediction(db, "dry")
    response = feedback(client, 1, "lava")
    assert response.status_code == 422 and response.json()["field"] == "actual"


def test_accuracy_empty(client, db):
    seed_prediction(db, "dry")
    assert client.get(ACCURACY).json() == {"with_feedback": 0, "correct": 0, "accuracy": None}


def test_accuracy(client, db):
    for surface, actual in [("dry", "dry"), ("wet", "wet"), ("snow", "snow"), ("wet", "snow"), ("dirt", None)]:
        seed_prediction(db, surface, actual)
    assert client.get(ACCURACY).json() == {"with_feedback": 4, "correct": 3, "accuracy": 0.75}


def test_accuracy_rounding(client, db):
    for surface, actual in [("dry", "dry"), ("wet", "wet"), ("wet", "dirt")]:
        seed_prediction(db, surface, actual)
    assert client.get(ACCURACY).json()["accuracy"] == 0.667


def test_accuracy_one_query(client, db, monkeypatch):
    for surface, actual in [("dry", "dry"), ("wet", "snow")]:
        seed_prediction(db, surface, actual)
    calls = []
    original = psycopg.Connection.execute

    def counting(self, *args, **kwargs):
        calls.append(args[0] if args else "")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(psycopg.Connection, "execute", counting)
    body = client.get(ACCURACY).json()
    assert body["accuracy"] == 0.5
    assert len(calls) == 1, "доля верных прогнозов считается одним запросом SQL"
