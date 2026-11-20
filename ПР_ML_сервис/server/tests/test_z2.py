"""З2. Ручная проверка признаков: validate_features в app/features.py. База данных этим тестам не нужна."""

from conftest import SAMPLE, sample, without

from app.features import FEATURE_NAMES, validate_features


def fields(errors):
    return [error["field"] for error in errors]


def test_valid_sample():
    features, errors = validate_features(SAMPLE)
    assert errors == []
    assert set(features) == set(FEATURE_NAMES) and len(features) == 14
    assert features["humidity"] == 63 and isinstance(features["humidity"], float), "числа нужно привести к float"
    assert features["tires"] == "летние"


def test_extra_key_is_ignored():
    features, errors = validate_features(sample(rec_speed=90))
    assert errors == [] and "rec_speed" not in features


def test_missing_feature():
    features, errors = validate_features(without("speed"))
    assert features is None
    assert fields(errors) == ["speed"]


def test_string_instead_of_number():
    features, errors = validate_features(sample(slip="2.12"))
    assert features is None and fields(errors) == ["slip"]


def test_bool_is_not_a_number():
    _, errors = validate_features(sample(humidity=True))
    assert fields(errors) == ["humidity"], "True – не число, хотя в Python bool – подкласс int"


def test_out_of_range():
    for name, value in [("air_temp", 150), ("road_temp", -999), ("lidar_intensity", 1.2), ("speed", -1)]:
        _, errors = validate_features(sample(**{name: value}))
        assert fields(errors) == [name], f"{name} = {value} должно быть ошибкой"


def test_bounds_are_allowed():
    _, errors = validate_features(sample(lidar_intensity=0, camera_brightness=255, slip=50))
    assert errors == []


def test_wrong_category():
    for name, value in [("tires", "шипы"), ("time_of_day", "Ночь"), ("wipers", 1)]:
        _, errors = validate_features(sample(**{name: value}))
        assert fields(errors) == [name], f"{name} = {value!r} должно быть ошибкой"


def test_all_errors_in_order():
    data = without("speed")
    data.update(slip="много", tires="шипы")
    features, errors = validate_features(data)
    assert features is None
    assert fields(errors) == ["speed", "slip", "tires"], "нужны все ошибки сразу, по порядку признаков"
    for error in errors:
        assert set(error) == {"field", "error"} and error["error"], "ошибка – словарь с полями field и error"


def test_not_a_dict():
    features, errors = validate_features([1, 2, 3])
    assert features is None and len(errors) == 1
