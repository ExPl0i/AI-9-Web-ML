"""Готовый модуль: коды и названия типов дорожного покрытия (из Т4.Б1.2–Т4.Б1.4)."""

from enum import StrEnum


class Surface(StrEnum):
    """Допустимые коды покрытия."""
    dry = "dry"
    wet = "wet"
    dirt = "dirt"
    snow = "snow"


SURFACE_NAMES = {"dry": "сухой асфальт", "wet": "мокрый асфальт", "dirt": "грунт", "snow": "снег и лед"}
SURFACE_MU = {"dry": 0.7, "wet": 0.4, "dirt": 0.5, "snow": 0.15}
