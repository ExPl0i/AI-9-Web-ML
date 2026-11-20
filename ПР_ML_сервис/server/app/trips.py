"""Готовый модуль: журнал поездок из Т4.Б1.4 (адреса /api/trips).

Писать и менять этот файл не нужно. Маршруты собраны в объект APIRouter («группа маршрутов»):
файл main.py подключает его одной строкой app.include_router(router). Подробно о разбиении
приложения на модули – в Т4.Б3.2.
"""

import datetime
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from app.db import Conn
from app.surfaces import Surface

router = APIRouter(prefix="/api/trips", tags=["Поездки"])


class TripIn(BaseModel):
    """Поездка, которую присылает клиент."""
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"date": "2026-11-05", "distance": 12.5, "surface": "wet", "note": "дождь после обеда"}],
    })

    date: datetime.date = Field(description="Дата поездки в виде ГГГГ-ММ-ДД")
    distance: float = Field(gt=0, le=2000, description="Расстояние, км: больше 0 и не больше 2000")
    surface: Surface = Field(description="Покрытие: dry, wet, dirt или snow")
    note: str = Field(default="", max_length=200, description="Комментарий водителя, не длиннее 200 символов")


class TripOut(BaseModel):
    """Поездка в ответе сервера: без служебных полей."""
    id: int = Field(description="Номер поездки")
    date: datetime.date
    distance: float = Field(description="Расстояние, км")
    surface: Surface
    note: str
    created_at: datetime.datetime = Field(description="Когда сервер принял поездку")


class TripSort(StrEnum):
    """Порядок списка поездок. Минус перед именем поля – по убыванию."""
    id = "id"
    date = "date"
    date_desc = "-date"
    distance = "distance"
    distance_desc = "-distance"


class SurfaceSummary(BaseModel):
    """Сводка по одному покрытию."""
    surface: Surface
    trips: int = Field(description="Число поездок")
    avg_distance: float = Field(description="Средний пробег, км, с округлением до 0,1")


TripId = Annotated[int, Path(ge=1, description="Номер поездки, начиная с 1")]
NOT_FOUND = {404: {"description": "Поездки с таким номером нет"}}
TRIP_COLUMNS = "id, date, distance, surface, note, created_at"
ORDER_BY = {
    TripSort.id: "id",
    TripSort.date: "date, id",
    TripSort.date_desc: "date DESC, id DESC",
    TripSort.distance: "distance, id",
    TripSort.distance_desc: "distance DESC, id DESC",
}


def not_found(trip_id: int) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Поездки с номером {trip_id} нет")


@router.get("", response_model=list[TripOut], summary="Список поездок")
def list_trips(
    conn: Conn,
    surface: Annotated[Surface | None, Query(description="Только поездки по этому покрытию")] = None,
    min_distance: Annotated[float, Query(ge=0, description="Только поездки не короче, км")] = 0,
    sort: Annotated[TripSort, Query(description="Порядок: id, date, -date, distance, -distance")] = TripSort.id,
):
    """Поездки из таблицы trips с фильтрами и сортировкой."""
    conditions = ["distance >= %(min_distance)s"]
    params: dict = {"min_distance": min_distance}
    if surface is not None:
        conditions.append("surface = %(surface)s")
        params["surface"] = surface
    query = f"SELECT {TRIP_COLUMNS} FROM trips WHERE {' AND '.join(conditions)} ORDER BY {ORDER_BY[sort]}"
    return conn.execute(query, params).fetchall()


@router.post("", status_code=201, response_model=TripOut, summary="Добавить поездку")
def create_trip(trip: TripIn, conn: Conn, request: Request, response: Response):
    """Добавляет поездку одной командой INSERT ... RETURNING."""
    row = conn.execute(
        f"""INSERT INTO trips (date, distance, surface, note, client_host)
            VALUES (%(date)s, %(distance)s, %(surface)s, %(note)s, %(client_host)s)
            RETURNING {TRIP_COLUMNS}""",
        {**trip.model_dump(), "client_host": request.client.host if request.client else None},
    ).fetchone()
    response.headers["Location"] = f"{request.url.path}/{row['id']}"
    return row


@router.get("/summary", response_model=list[SurfaceSummary], summary="Сводка по покрытиям")
def trips_summary(conn: Conn):
    """Число поездок и средний пробег по каждому покрытию."""
    return conn.execute(
        """SELECT surface, COUNT(*) AS trips, ROUND(AVG(distance), 1) AS avg_distance
           FROM trips GROUP BY surface ORDER BY surface"""
    ).fetchall()


@router.get("/{trip_id}", response_model=TripOut, summary="Поездка по номеру", responses=NOT_FOUND)
def get_trip(trip_id: TripId, conn: Conn):
    row = conn.execute(f"SELECT {TRIP_COLUMNS} FROM trips WHERE id = %s", [trip_id]).fetchone()
    if row is None:
        raise not_found(trip_id)
    return row


@router.put("/{trip_id}", response_model=TripOut, summary="Заменить поездку", responses=NOT_FOUND)
def replace_trip(trip_id: TripId, trip: TripIn, conn: Conn):
    row = conn.execute(
        f"""UPDATE trips SET date = %(date)s, distance = %(distance)s, surface = %(surface)s, note = %(note)s
            WHERE id = %(id)s RETURNING {TRIP_COLUMNS}""",
        {**trip.model_dump(), "id": trip_id},
    ).fetchone()
    if row is None:
        raise not_found(trip_id)
    return row


@router.delete("/{trip_id}", status_code=204, summary="Удалить поездку", responses=NOT_FOUND)
def delete_trip(trip_id: TripId, conn: Conn):
    cursor = conn.execute("DELETE FROM trips WHERE id = %s", [trip_id])
    if cursor.rowcount == 0:
        raise not_found(trip_id)
    return Response(status_code=204)
