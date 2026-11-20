"""З5. Журнал прогнозов: таблица predictions в schema.sql и запись каждого прогноза."""

import psycopg
from conftest import EXAMPLES, SAMPLE, VERSION_1, check_model_loaded, sample, without
from fastapi.testclient import TestClient

import app.main as main

API = "/api/predict"


def columns(db) -> dict:
    rows = db.execute(
        """SELECT column_name, data_type, is_nullable, is_identity, identity_generation
           FROM information_schema.columns WHERE table_name = 'predictions'"""
    ).fetchall()
    return {row["column_name"]: row for row in rows}


def test_table_columns(db):
    cols = columns(db)
    assert cols, "в schema.sql нет таблицы predictions"
    assert cols["id"]["is_identity"] == "YES" and cols["id"]["identity_generation"] == "ALWAYS"
    assert cols["features"]["data_type"] == "jsonb"
    assert cols["probabilities"]["data_type"] == "jsonb"
    assert cols["surface"]["data_type"] == "text"
    assert cols["model_version"]["data_type"] == "text"
    assert cols["elapsed_ms"]["data_type"] in {"double precision", "numeric", "real"}
    assert cols["created_at"]["data_type"] == "timestamp with time zone"
    for name in ("features", "surface", "probabilities", "model_version", "created_at"):
        assert cols[name]["is_nullable"] == "NO", f"столбец {name} должен быть NOT NULL"
    assert cols["actual"]["is_nullable"] == "YES", "actual пустой, пока водитель не ответил"
    assert cols["feedback_at"]["data_type"] == "timestamp with time zone"


def test_prediction_is_logged(client, db):
    response = client.post(API, json=SAMPLE)
    check_model_loaded(response)
    body = response.json()
    rows = db.execute("SELECT * FROM predictions").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert body["id"] == row["id"] == 1
    assert row["features"] == SAMPLE, "в столбце features – все 14 признаков запроса"
    assert row["surface"] == body["surface"]
    assert row["probabilities"] == body["probabilities"]
    assert row["model_version"] == VERSION_1
    assert row["actual"] is None
    assert body["created_at"]


def test_numbering(client, db):
    ids = [client.post(API, json=example["features"]).json()["id"] for example in EXAMPLES[:3]]
    assert ids == [1, 2, 3]
    assert db.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"] == 3


def test_invalid_request_is_not_logged(client, db):
    assert client.post(API, json=without("tires")).status_code == 422
    assert client.post(API, json=sample(speed=-5)).status_code == 422
    assert db.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"] == 0


def test_jsonb_operators(client, db):
    client.post(API, json=sample(tires="зимние", air_temp=-3.5))
    row = db.execute("SELECT features ->> 'tires' AS tires, (features ->> 'air_temp')::numeric AS air "
                     "FROM predictions").fetchone()
    assert row["tires"] == "зимние" and float(row["air"]) == -3.5


def test_history_route(client):
    for example in EXAMPLES[:3]:
        client.post(API, json=example["features"])
    records = client.get("/api/predictions", params={"limit": 2}).json()
    assert [record["id"] for record in records] == [3, 2], "новые записи сначала"


def test_committed(real_db):
    with TestClient(main.app) as client:
        response = client.post(API, json=SAMPLE)
        check_model_loaded(response)
        assert response.status_code == 200, response.text
    with psycopg.connect(real_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1, "запись должна быть подтверждена"
