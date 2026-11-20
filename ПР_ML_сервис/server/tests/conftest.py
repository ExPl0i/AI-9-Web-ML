"""Общие помощники тестов. Этот файл писать не нужно, он готовый.

Тесты работают с тестовой базой pokrytie_test (строка TEST_DATABASE_URL в .env), рабочую базу не трогают.
Большинство тестов выполняются внутри транзакции, которая в конце отменяется (ROLLBACK):
после теста база остается такой же, как до него, и тесты не мешают друг другу.

Эталонные ответы модели лежат в tests/data/reference.json: десять наборов признаков и ответы двух версий
модели на них. Их записал скрипт model/train_model.py в момент обучения модели.
"""

import json

import joblib
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import BASE_DIR, settings
from app.db import get_conn

TEST_URL = settings.test_database_url
DATA_DIR = BASE_DIR / "tests" / "data"
REFERENCE = json.loads((DATA_DIR / "reference.json").read_text(encoding="utf-8"))
EXAMPLES = REFERENCE["examples"]
SAMPLE = EXAMPLES[0]["features"]          # 14 признаков без ошибок
VERSION_1 = "logreg-1.0"
VERSION_2 = "logreg-1.1-snow5"
CODES = {"сухой асфальт": "dry", "мокрый асфальт": "wet", "грунт": "dirt", "снег и лед": "snow"}
NAMES = {code: name for name, code in CODES.items()}
ORDER = ["dry", "wet", "dirt", "snow"]
TABLES = ("trips", "predictions")


def sample(**changes) -> dict:
    """Признаки SAMPLE с заменой некоторых значений."""
    return {**SAMPLE, **changes}


def without(*names) -> dict:
    """Признаки SAMPLE без указанных признаков."""
    return {key: value for key, value in SAMPLE.items() if key not in names}


def expected(example: dict, version: str = VERSION_1) -> tuple[str, dict[str, float]]:
    """Эталонный ответ: код покрытия и вероятности по кодам покрытия."""
    answer = example["expected"][version]
    return CODES[answer["class"]], {CODES[name]: p for name, p in answer["probabilities"].items()}


def close_probabilities(got: dict, want: dict, tolerance: float = 1e-3) -> bool:
    return set(got) == set(want) and all(abs(got[code] - want[code]) <= tolerance for code in want)


@pytest.fixture(scope="session")
def bundle():
    """Файл модели, загруженный напрямую через joblib (мимо вашей функции load_model)."""
    return joblib.load(settings.model_path)


def run_file(url, path):
    """Выполняет SQL-файл целиком в базе url и подтверждает изменения."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def schema():
    """Таблицы сервиса в тестовой базе: создаются заново по вашему schema.sql (задание З5)."""
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS trips, predictions, cars CASCADE")
    run_file(TEST_URL, BASE_DIR / "schema.sql")


def clean_tables(conn) -> None:
    """Очищает таблицы сервиса (только те, что уже есть) и сбрасывает счетчики номеров."""
    for table in TABLES:
        if conn.execute("SELECT to_regclass(%s) IS NOT NULL", [table]).fetchone()[0]:
            conn.execute(f"TRUNCATE {table} RESTART IDENTITY CASCADE")


@pytest.fixture
def db(schema):
    """Соединение с тестовой базой внутри транзакции: таблицы пустые, номера с 1, в конце ROLLBACK."""
    conn = psycopg.connect(TEST_URL)
    clean_tables(conn)
    conn.row_factory = dict_row
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def client(db):
    """Тестовый клиент: запросы приложения идут через соединение db (подмена зависимости get_conn).

    Блок with TestClient(...) запускает и функцию жизненного цикла lifespan: модель загружается при входе.
    """
    import app.main as main

    def test_conn():
        yield db

    main.app.dependency_overrides[get_conn] = test_conn
    with TestClient(main.app) as test_client:
        yield test_client
    main.app.dependency_overrides.clear()


@pytest.fixture
def real_db(schema, monkeypatch):
    """Приложение само открывает соединения к тестовой базе, изменения подтверждаются.

    До и после теста таблицы очищаются отдельной командой.
    """
    monkeypatch.setattr(settings, "database_url", TEST_URL)
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        clean_tables(conn)
    yield TEST_URL
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        clean_tables(conn)


def seed_prediction(db, surface: str, actual: str | None = None, features: dict | None = None,
                    version: str = VERSION_1) -> int:
    """Кладет прогноз прямо в таблицу predictions командой INSERT (в обход POST). Возвращает номер."""
    probabilities = {code: (0.7 if code == surface else 0.1) for code in ORDER}
    row = db.execute(
        """INSERT INTO predictions (features, surface, probabilities, model_version, elapsed_ms, actual)
           VALUES (%s, %s, %s, %s, 1.5, %s) RETURNING id""",
        [Jsonb(features or SAMPLE), surface, Jsonb(probabilities), version, actual],
    ).fetchone()
    return row["id"]


def check_model_loaded(response) -> None:
    """Понятное сообщение, если модель не загружена: сначала нужно сделать З1."""
    assert response.status_code != 503, "модель не загружена: сначала выполните задание З1 (функция lifespan)"
