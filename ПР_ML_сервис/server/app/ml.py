"""Т4.Б1.5, задания З1 и З4. Загрузка модели и прогноз (заготовка).

Файл модели model/model.joblib – словарь, сохраненный функцией joblib.dump (см. model/train_model.py):
    "pipeline"         конвейер scikit-learn: подготовка признаков и логистическая регрессия
    "version"          версия модели, например "logreg-1.0"
    "features"         список из 14 признаков в том порядке, в котором их видел конвейер при обучении
    "classes"          классы модели (названия покрытий на русском языке)
    "sklearn_version"  версия scikit-learn, которой модель обучена и сохранена
    "metrics", "trained_at", "train_rows", "description", "python_version" – паспорт модели

Внимание: joblib.load, как и pickle.load, при загрузке может выполнить произвольный код,
записанный в файле. Загружайте только модели, которые сохранили вы или ваш преподаватель.
"""

import pathlib
import time

import joblib
import pandas as pd
import sklearn

REQUIRED_KEYS = {"pipeline", "version", "features", "classes", "sklearn_version"}

# Модель отвечает названием покрытия на русском языке, а программный интерфейс – кодом покрытия
CLASS_TO_SURFACE = {"сухой асфальт": "dry", "мокрый асфальт": "wet", "грунт": "dirt", "снег и лед": "snow"}
SURFACE_ORDER = ["dry", "wet", "dirt", "snow"]   # порядок вероятностей в ответе


# ---------------------------------------------------------------- З1. Загрузка модели

def load_model(path) -> dict:
    """Загружает файл модели и проверяет, что им можно пользоваться.

    Ошибки:
        FileNotFoundError – файла нет (в тексте ошибки – полный путь);
        RuntimeError – в файле нет нужных ключей или модель сохранена другой версией scikit-learn.
    """
    # <-- З1. Напишите функцию:
    # 1. Превратите path в pathlib.Path (так функция примет и строку, и путь).
    # 2. Если файла нет (path.exists()), вызовите FileNotFoundError с полным путем в тексте ошибки.
    # 3. bundle = joblib.load(path)
    # 4. Если в словаре нет каких-то ключей из REQUIRED_KEYS (вычитание множеств: REQUIRED_KEYS - set(bundle)),
    #    вызовите RuntimeError и перечислите недостающие ключи.
    # 5. Если bundle["sklearn_version"] не равна sklearn.__version__, вызовите RuntimeError,
    #    в тексте которой есть обе версии и совет: установить версию из requirements.txt
    #    или переобучить модель командой python model/train_model.py.
    # 6. Верните bundle.
    raise NotImplementedError("З1: напишите load_model в app/ml.py")


# ---------------------------------------------------------------- З4. Прогноз

def predict_one(bundle: dict, features: dict) -> tuple[str, dict[str, float]]:
    """Прогноз для одного набора признаков.

    features – словарь из 14 признаков (уже проверенных моделью Features).
    Возвращает пару (код покрытия, вероятности по кодам покрытия), например
        ("snow", {"dry": 0.0012, "wet": 0.0841, "dirt": 0.0003, "snow": 0.9144}).
    Вероятности округлены до 4 знаков и идут в порядке SURFACE_ORDER.
    """
    # <-- З4. Напишите функцию:
    # 1. Таблица pandas из одной строки с теми же столбцами, что при обучении:
    #        frame = pd.DataFrame([features], columns=bundle["features"])
    #    ColumnTransformer в конвейере выбирает столбцы по именам, поэтому список чисел не подойдет.
    # 2. Вероятности: bundle["pipeline"].predict_proba(frame)[0] – массив из четырех чисел.
    #    Их порядок задает bundle["pipeline"].classes_ (названия классов по алфавиту), а не SURFACE_ORDER!
    # 3. Переведите названия классов в коды покрытия словарем CLASS_TO_SURFACE (zip по classes_ и вероятностям).
    # 4. Самый вероятный код – ответ модели.
    # 5. Верните (код, {код: round(вероятность, 4)}) с ключами в порядке SURFACE_ORDER.
    #    Вероятности приведите к float: numpy.float64 тоже работает, но в JSON лучше обычные числа.
    raise NotImplementedError("З4: напишите predict_one в app/ml.py")


def timed_predict(bundle: dict, features: dict) -> tuple[str, dict[str, float], float]:
    """predict_one и время прогноза в миллисекундах (time.perf_counter – часы для замеров)."""
    start = time.perf_counter()
    surface, probabilities = predict_one(bundle, features)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return surface, probabilities, round(elapsed_ms, 2)
