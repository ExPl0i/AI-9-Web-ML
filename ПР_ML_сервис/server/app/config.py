"""Готовый модуль: настройки приложения из файла .env.

Писать и менять этот файл не нужно.

Строки подключения к базе данных содержат пароль, поэтому их нет в коде и в Git.
Пути к файлам моделей имеют значения по умолчанию (папка server/model), их можно заменить в .env.
Они лежат в файле server/.env, а класс Settings из библиотеки pydantic-settings читает их оттуда
(переменные окружения с теми же именами имеют приоритет над файлом). Образец – файл .env.example.
Подробно о настройках через pydantic-settings – в Т4.Б3.2.
"""

import pathlib

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent   # папка server
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Настройки сервиса. Имя поля = имя переменной в .env (без учета регистра)."""
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(description="Строка подключения к рабочей базе pokrytie")
    test_database_url: str = Field(description="Строка подключения к тестовой базе pokrytie_test")
    model_path: pathlib.Path = Field(default=BASE_DIR / "model" / "model.joblib",
                                     description="Файл модели, которую сервис загружает при старте")
    model_v2_path: pathlib.Path = Field(default=BASE_DIR / "model" / "model_v2.joblib",
                                        description="Вторая версия модели (повышенная сложность)")


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        missing = ", ".join(str(e["loc"][0]).upper() for e in error.errors())
        raise RuntimeError(
            f"Не заданы настройки {missing}. Скопируйте файл .env.example в .env (папка server) "
            "и впишите в строки подключения свой пароль."
        ) from None


settings = load_settings()
