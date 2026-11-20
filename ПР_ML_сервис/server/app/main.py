"""Т4.Б1.5. Модель машинного обучения как веб-сервис (заготовка).

Модель определения покрытия из итоговой работы Темы 1 отвечает на запросы по сети,
проверяет входные данные и записывает каждый прогноз в журнал в базе данных PostgreSQL.
Запуск (в папке server, сервер PostgreSQL запущен, таблицы созданы командой python -m app.init_db):
    fastapi dev app/main.py
затем в браузере http://127.0.0.1:8000/docs.

Задания:
    З1  загрузка модели один раз при старте: load_model в app/ml.py и функция lifespan ниже
    З2  ручная проверка признаков validate_features в app/features.py
    З3  модель Pydantic Features в app/features.py вместо ручной проверки
    З4  прогноз predict_one в app/ml.py и маршрут POST /api/predict с моделью ответа PredictionOut
    З5  таблица predictions в schema.sql и запись каждого прогноза в журнал
    З6  обратная связь POST /api/predictions/{prediction_id}/feedback и доля верных прогнозов

Повышенная сложность (pytest -m extra): пакетный прогноз по файлу CSV, поиск по признаку внутри jsonb,
сравнение двух версий модели.
"""

import collections
import csv
import datetime
import io
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Annotated

from fastapi import FastAPI, HTTPException, Path, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import BASE_DIR, settings
from app.db import Conn
from app.errors import NOT_JSON, describe_all, http_error_text, is_json_error
from app.features import FEATURE_NAMES, NUMERIC_SPECS, Features
from app.ml import CLASS_TO_SURFACE, load_model, predict_one, timed_predict
from app.offline_docs import DOCS_SETTINGS, add_offline_docs
from app.surfaces import SURFACE_MU, SURFACE_NAMES, Surface
from app.trips import router as trips_router

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

TAGS = [
    {"name": "Служебное", "description": "Проверка связи с базой данных"},
    {"name": "Модель", "description": "Прогноз покрытия по показаниям датчиков"},
    {"name": "Журнал прогнозов", "description": "Прогнозы в базе данных и обратная связь от водителей"},
    {"name": "Покрытия", "description": "Справочник типов дорожного покрытия"},
    {"name": "Поездки", "description": "Журнал поездок (готов из Т4.Б1.4)"},
]


# ---------------------------------------------------------------- З1. Загрузка модели при старте

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Жизненный цикл приложения: код до yield выполняется один раз при старте сервера,
    код после yield – один раз при остановке.

    Модель загружается здесь, а не в обработчике: загрузка файла занимает десятки миллисекунд,
    а прогноз – единицы. app.state – общее хранилище приложения, обработчики берут модель оттуда.
    """
    # <-- З1. До yield загрузите модель функцией load_model(settings.model_path)
    #     и положите результат в app.state.model. Обработчики берут модель оттуда (функция current_model).
    #     Вызывайте load_model именно по этому имени: тест проверяет, сколько раз она вызвана.
    yield


app = FastAPI(
    title="Покрытие",
    version="0.5.0",
    description="Программный интерфейс учебного сервиса беспилотного автомобиля: прогноз типа покрытия "
                "по показаниям датчиков и журнал прогнозов в PostgreSQL. Занятие Т4.Б1.5.",
    openapi_tags=TAGS,
    lifespan=lifespan,
    **DOCS_SETTINGS,
)
add_offline_docs(app)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.include_router(trips_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


def current_model(request: Request, name: str = "model") -> dict:
    """Модель из app.state. Если lifespan ее не загрузил, ответ 503 «сервис недоступен»."""
    model = getattr(request.app.state, name, None)
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена: напишите функцию lifespan (задание З1)")
    return model


# ---------------------------------------------------------------- Модели ответов

class ModelInfo(BaseModel):
    """Паспорт модели, которая сейчас отвечает на запросы."""
    version: str = Field(description="Версия модели")
    description: str = Field(description="Что это за модель")
    trained_at: str = Field(description="Дата обучения")
    train_rows: int = Field(description="Сколько строк было в обучающих данных")
    features: list[str] = Field(description="Признаки в том порядке, в котором их видела модель при обучении")
    classes: list[Surface] = Field(description="Коды покрытий, которые различает модель")
    metrics: dict[str, float] = Field(description="Качество модели при обучении: F-мера, полнота класса «снег и лед»")
    sklearn_version: str = Field(description="Версия scikit-learn, которой модель сохранена")


class PredictionOut(BaseModel):
    """Ответ сервиса на запрос прогноза."""
    # <-- З5. Добавьте поле id: int – номер записи в журнале прогнозов
    surface: Surface = Field(description="Самое вероятное покрытие")
    surface_name: str = Field(description="Название покрытия на русском языке")
    probabilities: dict[str, float] = Field(description="Вероятности всех четырех покрытий, сумма равна 1")
    model_version: str = Field(description="Версия модели, которая дала ответ")
    elapsed_ms: float = Field(description="Время прогноза, мс")
    # <-- З5. Добавьте поле created_at: datetime.datetime – когда прогноз записан в журнал


class PredictionRecord(BaseModel):
    """Запись журнала прогнозов."""
    id: int
    created_at: datetime.datetime
    features: dict = Field(description="Признаки, по которым сделан прогноз")
    surface: Surface
    probabilities: dict[str, float]
    model_version: str
    elapsed_ms: float
    actual: Surface | None = Field(description="Настоящее покрытие по словам водителя; null, если отметки нет")


class FeedbackIn(BaseModel):
    """Обратная связь: какое покрытие было на самом деле."""
    actual: Surface = Field(description="Настоящее покрытие: dry, wet, dirt или snow")


class FeedbackOut(BaseModel):
    id: int
    surface: Surface = Field(description="Ответ модели")
    actual: Surface = Field(description="Настоящее покрытие")
    correct: bool = Field(description="Совпал ли ответ модели с настоящим покрытием")


class AccuracyOut(BaseModel):
    with_feedback: int = Field(description="Сколько прогнозов получили обратную связь")
    correct: int = Field(description="Сколько из них верные")
    accuracy: float | None = Field(description="Доля верных прогнозов (до 0,001); null, если обратной связи нет")


PredictionId = Annotated[int, Path(ge=1, description="Номер прогноза в журнале")]
RECORD_COLUMNS = "id, created_at, features, surface, probabilities, model_version, elapsed_ms, actual"


# ---------------------------------------------------------------- Готовые маршруты

@app.get("/api/health", tags=["Служебное"], summary="Связь с базой данных")
def health(conn: Conn) -> dict:
    """Проверяет, что приложение подключается к базе: версия сервера, база и пользователь."""
    row = conn.execute("SELECT version() AS version, current_database() AS database, current_user AS user").fetchone()
    return {"database": row["database"], "user": row["user"], "server": row["version"].split(",")[0]}


@app.get("/api/surfaces", tags=["Покрытия"], summary="Список покрытий")
def list_surfaces() -> list[dict]:
    """Четыре типа покрытия: код, название и учебный коэффициент сцепления."""
    return [{"code": code, "name": name, "mu": SURFACE_MU[code]} for code, name in SURFACE_NAMES.items()]


@app.get("/api/model", response_model=ModelInfo, tags=["Модель"], summary="Паспорт модели")
def model_info(request: Request):
    """Какая модель сейчас отвечает на запросы: версия, признаки, классы, качество, версия scikit-learn."""
    bundle = current_model(request)
    return {
        "version": bundle["version"],
        "description": bundle.get("description", ""),
        "trained_at": bundle.get("trained_at", ""),
        "train_rows": bundle.get("train_rows", 0),
        "features": bundle["features"],
        "classes": [CLASS_TO_SURFACE[name] for name in bundle["classes"]],
        "metrics": {key: value for key, value in bundle.get("metrics", {}).items() if key != "C"},
        "sklearn_version": bundle["sklearn_version"],
    }


@app.get("/api/predictions", response_model=list[PredictionRecord], tags=["Журнал прогнозов"],
         summary="Последние прогнозы")
def list_predictions(
    conn: Conn,
    limit: Annotated[int, Query(ge=1, le=100, description="Сколько последних записей вернуть")] = 20,
):
    """Последние записи журнала, новые сначала. Столбцы jsonb psycopg сам превращает в словари Python."""
    return conn.execute(f"SELECT {RECORD_COLUMNS} FROM predictions ORDER BY id DESC LIMIT %s", [limit]).fetchall()


# ---------------------------------------------------------------- З4 и З5. Прогноз и запись в журнал

@app.post("/api/predict", response_model=PredictionOut, tags=["Модель"], summary="Прогноз покрытия")
def predict(features: Features, request: Request, conn: Conn):
    """Прогноз покрытия по 14 признакам. Каждый прогноз записывается в журнал predictions.

    Входные данные проверяет модель Features: ошибки приходят ответом 422 с именем признака.
    """
    # <-- З4. Напишите обработчик:
    # 1. bundle = current_model(request) – модель, которую загрузил lifespan.
    # 2. data = features.model_dump(mode="json") – словарь признаков; mode="json" превращает перечисления
    #    в обычные строки ("выкл"), а не в объекты Wipers.
    # 3. surface, probabilities, elapsed_ms = timed_predict(bundle, data) – прогноз и время в миллисекундах.
    # 4. Верните словарь с полями PredictionOut: surface, surface_name (SURFACE_NAMES[surface]),
    #    probabilities, model_version (bundle["version"]), elapsed_ms.
    #
    # <-- З5. Перед ответом запишите прогноз в таблицу predictions одной командой
    #    INSERT INTO predictions (features, surface, probabilities, model_version, elapsed_ms)
    #    VALUES (%(features)s, ...) RETURNING id, created_at
    #    Словари data и probabilities передавайте в обертке Jsonb(...): так psycopg знает, что это jsonb.
    #    Номер и время из RETURNING добавьте в ответ.
    raise NotImplementedError("З4: напишите обработчик POST /api/predict в app/main.py")


# ---------------------------------------------------------------- З6. Обратная связь и доля верных

@app.get("/api/predictions/accuracy", response_model=AccuracyOut, tags=["Журнал прогнозов"],
         summary="Доля верных прогнозов")
def accuracy(conn: Conn):
    """Доля верных прогнозов среди тех, по которым водители сообщили настоящее покрытие. Один запрос SQL."""
    # <-- З6. Один запрос SQL к таблице predictions, который возвращает три столбца:
    #   with_feedback – сколько прогнозов с обратной связью: COUNT(actual) считает только непустые значения;
    #   correct       – сколько из них верные: COUNT(*) FILTER (WHERE actual = surface);
    #   accuracy      – correct / with_feedback, округлить до 3 знаков. Деление на ноль, когда обратной связи
    #                   нет, обходите функцией NULLIF(COUNT(actual), 0): тогда результат NULL (в ответе null).
    #                   Не забудьте ::numeric у делимого: деление целых чисел в SQL отбрасывает дробную часть.
    # Верните conn.execute(...).fetchone().
    raise NotImplementedError("З6: напишите долю верных прогнозов в app/main.py")


@app.post("/api/predictions/{prediction_id}/feedback", response_model=FeedbackOut, tags=["Журнал прогнозов"],
          summary="Обратная связь от водителя", responses={404: {"description": "Прогноза с таким номером нет"}})
def feedback(prediction_id: PredictionId, body: FeedbackIn, conn: Conn):
    """Водитель сообщает настоящее покрытие. Повторная отметка заменяет прежнюю."""
    # <-- З6. Одна команда UPDATE predictions ... SET actual = ..., feedback_at = now() WHERE id = ...
    #    RETURNING id, surface, actual, surface = actual AS correct
    #    Код покрытия передавайте как body.actual.value. Если строки с таким номером нет (fetchone() вернул None),
    #    вызовите HTTPException 404 с текстом «Прогноза с номером ... нет».
    raise NotImplementedError("З6: напишите обработчик обратной связи в app/main.py")


# ---------------------------------------------------------------- Повышенная сложность (pytest -m extra)

# 1. POST /api/predict/batch – пакетный прогноз по файлу CSV (параметр file: UploadFile).
#    Прочитайте файл (file.file.read().decode("utf-8-sig")), разберите csv.DictReader, оставьте в каждой строке
#    только столбцы из FEATURE_NAMES и проверьте ее Features.model_validate(row, strict=False): в CSV все
#    значения – строки. Ответ: {"rows": число строк, "counts": {код: число прогнозов},
#    "predictions": [{"line": номер строки файла, "surface": ..., "probabilities": ...}],
#    "errors": [{"line": ..., "field": ..., "error": ...}]}. Первая строка файла – заголовок, данные с line = 2.
# 2. GET /api/predictions/search?feature=air_temp&min=-10&max=0 – прогнозы, у которых числовой признак
#    внутри столбца jsonb лежит в границах: (features ->> %(feature)s::text)::numeric. Допустимые имена признака
#    задайте перечислением из NUMERIC_SPECS, неизвестный признак – ответ 422. Ответ – list[PredictionRecord].
# 3. POST /api/predict/compare – ответы двух версий модели на одни признаки:
#    [{"model_version": ..., "surface": ..., "probabilities": ...}, ...]. Вторую версию загрузите в lifespan
#    из settings.model_v2_path в app.state.model_v2.


# ---------------------------------------------------------------- Единый формат ошибки (готово из Т4.Б1.3)

@app.exception_handler(RequestValidationError)
def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if is_json_error(errors):
        return JSONResponse({"error": NOT_JSON, "field": None}, status_code=400)
    problems = describe_all(errors)
    return JSONResponse({"error": problems[0]["error"], "field": problems[0]["field"], "errors": problems},
                        status_code=422)


@app.exception_handler(StarletteHTTPException)
def http_error_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": http_error_text(exc.detail, request.method), "field": None},
                        status_code=exc.status_code, headers=exc.headers)


# ---------------------------------------------------------------- Политика CORS (готово из Т4.Б1.3)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["Location"],
)
