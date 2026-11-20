"""З3. Модель Pydantic Features вместо ручной проверки: ошибки 422 и описания признаков в документации."""

import pytest
from conftest import SAMPLE, sample, without
from pydantic import ValidationError

from app.features import FEATURE_NAMES, Features

API = "/api/predict"


def test_model_fields():
    assert list(Features.model_fields) == FEATURE_NAMES, "14 полей в порядке FEATURE_NAMES"
    assert Features(**SAMPLE).humidity == 63


@pytest.mark.parametrize("change", [{"speed": "44.8"}, {"speed": True}, {"speed": 999}, {"tires": "шипы"},
                                    {"rec_speed": 90}])
def test_model_rejects(change):
    with pytest.raises(ValidationError):
        Features.model_validate(sample(**change))


def test_missing_feature_422(client):
    response = client.post(API, json=without("speed"))
    assert response.status_code == 422
    assert response.json()["field"] == "speed"


def test_string_instead_of_number_422(client):
    body = client.post(API, json=sample(slip="2.12")).json()
    assert body["field"] == "slip" and body["error"] == "должно быть числом"


def test_out_of_range_422(client):
    body = client.post(API, json=sample(air_temp=150)).json()
    assert body["field"] == "air_temp" and "60" in body["error"]


def test_wrong_category_422(client):
    body = client.post(API, json=sample(tires="шипы")).json()
    assert body["field"] == "tires" and "летние" in body["error"]


def test_extra_feature_422(client):
    response = client.post(API, json=sample(rec_speed=90))
    assert response.status_code == 422, "лишний признак rec_speed должен давать ошибку (extra='forbid')"
    assert response.json()["field"] == "rec_speed"


def test_all_errors_listed(client):
    data = without("speed")
    data.update(slip="много", tires="шипы")
    body = client.post(API, json=data).json()
    assert {error["field"] for error in body["errors"]} == {"speed", "slip", "tires"}


def test_descriptions_in_openapi(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["Features"]
    assert set(schema["properties"]) == set(FEATURE_NAMES)
    assert set(schema["required"]) == set(FEATURE_NAMES), "все 14 признаков обязательные"
    assert schema.get("additionalProperties") is False, "лишние поля запрещены"
    for name, prop in schema["properties"].items():
        assert prop.get("description"), f"у признака {name} нет описания"
    assert schema["properties"]["speed"]["minimum"] == 0 and schema["properties"]["speed"]["maximum"] == 200
    assert schema["properties"]["slip"]["maximum"] == 50
    assert schema["properties"]["air_temp"]["minimum"] == -60
