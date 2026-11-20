"""Т4.Б1.5. Обучение модели для сервиса «Покрытие» (инструмент преподавателя).

Ученикам запускать этот файл не нужно: готовые модели model.joblib и model_v2.joblib лежат рядом.
Скрипт повторяет решение итоговой работы Темы 1 (вариант 1) и сохраняет конвейер scikit-learn
вместе с паспортом модели: версией, списком признаков, классами, версиями библиотек и метриками.

Когда запускать: если на школьных компьютерах стоит другая версия scikit-learn, чем в requirements.txt.
Сервис откажется загружать модель, сохраненную другой версией, поэтому модель нужно переобучить
той версией, которая установлена. Скрипт заодно пересчитывает эталонные ответы для тестов
(tests/data/reference.json).

Запуск из папки server:
    python model/train_model.py
"""

import datetime
import json
import pathlib
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = pathlib.Path(__file__).resolve().parent
SERVER_DIR = MODEL_DIR.parent
DATA_FILE = MODEL_DIR / "road_surface_v01.csv"
REFERENCE_FILE = SERVER_DIR / "tests" / "data" / "reference.json"
RANDOM_STATE = 42

# Признаки, которые модель получает на входе. trip_id (служебный номер) и rec_speed (утечка ответа:
# рекомендованную скорость система выставляет уже после распознавания покрытия) исключены,
# как в итоговой работе Темы 1.
NUM = ["speed", "vib_front", "vib_rear", "vib_hf", "slip", "air_temp", "road_temp", "humidity",
       "lidar_intensity", "camera_brightness", "cabin_noise"]
CAT = ["wipers", "time_of_day", "tires"]
FEATURES = NUM + CAT
SNOW = "снег и лед"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка из итоговой работы: дубликаты, разнобой в записи, коды ошибок датчиков."""
    df = df.drop_duplicates().copy()
    df["time_of_day"] = df["time_of_day"].str.strip().str.lower()
    df.loc[df["road_temp"] < -100, "road_temp"] = np.nan
    df.loc[(df["air_temp"] < -60) | (df["air_temp"] > 60), "air_temp"] = np.nan
    df.loc[df["slip"] > 50, "slip"] = np.nan
    return df


def make_pipeline_lr(class_weight=None) -> Pipeline:
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CAT),
    ])
    return Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, class_weight=class_weight))])


def scores(pipe, X, y) -> dict:
    pred = pipe.predict(X)
    return {"f1_macro": round(float(f1_score(y, pred, average="macro")), 3),
            "recall_snow": round(float(recall_score(y, pred, labels=[SNOW], average="macro")), 3)}


def train(class_weight, X_train, y_train, X_test, y_test, X_all, y_all, cv):
    """Подбор C кросс-валидацией, проверка на тесте, затем обучение на всех данных."""
    search = GridSearchCV(make_pipeline_lr(class_weight), {"model__C": [0.01, 0.1, 1, 10, 100]},
                          cv=cv, scoring="f1_macro").fit(X_train, y_train)
    best_c = search.best_params_["model__C"]
    test = scores(search.best_estimator_, X_test, y_test)
    final = make_pipeline_lr(class_weight).set_params(model__C=best_c).fit(X_all, y_all)
    metrics = {"C": best_c, "cv_f1_macro": round(float(search.best_score_), 3),
               "test_f1_macro": test["f1_macro"], "test_recall_snow": test["recall_snow"]}
    return final, metrics


def bundle(pipeline, version, description, metrics, n_rows) -> dict:
    """Паспорт модели: все, что сервису нужно знать о файле, кроме самого конвейера."""
    return {
        "pipeline": pipeline,
        "version": version,
        "description": description,
        "features": FEATURES,
        "classes": [str(c) for c in pipeline.classes_],
        "trained_at": datetime.date.today().isoformat(),
        "train_rows": n_rows,
        "metrics": metrics,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }


def reference_examples(df: pd.DataFrame) -> list[dict]:
    """Десять примеров без пропусков для тестов: по два-три на каждый класс."""
    full = df.dropna(subset=FEATURES)
    picked = []
    for surface, count in [("сухой асфальт", 3), ("мокрый асфальт", 3), ("грунт", 2), (SNOW, 2)]:
        rows = full[full["surface"] == surface].sample(count, random_state=RANDOM_STATE)
        picked.append(rows)
    rows = pd.concat(picked)
    examples = []
    for _, row in rows.iterrows():
        features = {}
        for name in NUM:
            value = float(row[name])
            features[name] = int(value) if value.is_integer() else value
        for name in CAT:
            features[name] = str(row[name])
        examples.append({"features": features, "label": str(row["surface"])})
    return examples


def main() -> None:
    df = clean(pd.read_csv(DATA_FILE))
    X, y = df[FEATURES], df["surface"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    v1, m1 = train(None, X_train, y_train, X_test, y_test, X, y, cv)
    weights = {"сухой асфальт": 1, "мокрый асфальт": 1, "грунт": 1, SNOW: 5}
    v2, m2 = train(weights, X_train, y_train, X_test, y_test, X, y, cv)

    joblib.dump(bundle(v1, "logreg-1.0", "Логистическая регрессия из итоговой работы Темы 1, вариант 1", m1, len(df)),
                MODEL_DIR / "model.joblib")
    joblib.dump(bundle(v2, "logreg-1.1-snow5", "То же, вес класса «снег и лед» увеличен в 5 раз", m2, len(df)),
                MODEL_DIR / "model_v2.joblib")
    print("версия 1:", m1)
    print("версия 2:", m2)

    examples = reference_examples(df)
    frame = pd.DataFrame([e["features"] for e in examples], columns=FEATURES)
    reference = []
    for model, version in [(v1, "logreg-1.0"), (v2, "logreg-1.1-snow5")]:
        proba = model.predict_proba(frame)
        for example, p in zip(examples, proba):
            example.setdefault("expected", {})[version] = {
                "class": str(model.classes_[int(np.argmax(p))]),
                "probabilities": {str(c): round(float(v), 6) for c, v in zip(model.classes_, p)},
            }
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    REFERENCE_FILE.write_text(json.dumps({"sklearn_version": sklearn.__version__, "examples": examples},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print("эталон:", REFERENCE_FILE, "примеров:", len(examples))


if __name__ == "__main__":
    main()
