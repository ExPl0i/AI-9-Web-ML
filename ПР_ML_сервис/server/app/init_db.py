"""Готовый модуль (из Т4.Б1.4): создание таблиц по файлу schema.sql.

Запуск из папки server:
    python -m app.init_db          таблицы в рабочей базе (DATABASE_URL)
    python -m app.init_db --test   таблицы в тестовой базе (TEST_DATABASE_URL)
"""

import sys

import psycopg

from app.config import BASE_DIR, settings

SCHEMA_FILE = BASE_DIR / "schema.sql"


def create_tables(url: str) -> None:
    """Выполняет все команды schema.sql в базе по строке подключения url."""
    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    with psycopg.connect(url) as conn:
        conn.execute(schema)


def main() -> None:
    test = "--test" in sys.argv
    url = settings.test_database_url if test else settings.database_url
    create_tables(url)
    print("Таблицы созданы в базе", "pokrytie_test" if test else "pokrytie")


if __name__ == "__main__":
    main()
