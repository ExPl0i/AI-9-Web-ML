"""Готовый модуль (из Т4.Б1.4): соединение с базой данных PostgreSQL – одно на запрос."""

from typing import Annotated

import psycopg
from fastapi import Depends
from psycopg.rows import dict_row

from app.config import settings


def get_conn():
    """Зависимость FastAPI: открывает соединение с базой на время обработки запроса.

    Блок with у соединения psycopg в конце подтверждает изменения (COMMIT),
    при исключении внутри блока отменяет их (ROLLBACK), а затем закрывает соединение.
    row_factory=dict_row: строки результата приходят словарями {"имя столбца": значение}.
    """
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        yield conn


# Тип параметра для обработчиков: conn: Conn.
# scope="function": соединение закрывается (и изменения подтверждаются) после выполнения
# функции маршрута, но ДО отправки ответа клиенту. Иначе клиент мог бы получить ответ 201
# раньше, чем запись будет подтверждена в базе.
Conn = Annotated[psycopg.Connection, Depends(get_conn, scope="function")]
