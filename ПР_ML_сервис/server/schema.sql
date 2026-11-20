-- Т4.Б1.5. Схема базы данных сервиса «Покрытие».
-- Создает таблицы, если их еще нет: python -m app.init_db (рабочая база) или тесты (тестовая база).

-- Журнал поездок (готово из Т4.Б1.4)
CREATE TABLE IF NOT EXISTS trips (
    id          integer     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,  -- номер выдает сама база
    date        date        NOT NULL,                                   -- дата поездки
    distance    numeric     NOT NULL,                                   -- расстояние, км (точное десятичное число)
    surface     text        NOT NULL,                                   -- код покрытия: dry, wet, dirt, snow
    note        text        NOT NULL DEFAULT '',                        -- комментарий водителя
    created_at  timestamptz NOT NULL DEFAULT now(),                     -- когда сервер принял поездку
    client_host text                                                    -- служебное поле: адрес клиента
);

-- З5. Журнал прогнозов модели.  <-- напишите CREATE TABLE IF NOT EXISTS predictions со столбцами:
--   id             номер, который выдает база (как в trips)
--   created_at     дата и время прогноза с часовым поясом, по умолчанию now(), обязательный
--   features       все 14 признаков одним значением типа jsonb, обязательный
--   surface        ответ модели: код покрытия (text), обязательный
--   probabilities  вероятности всех классов, jsonb, обязательный
--   model_version  версия модели (text), обязательный
--   elapsed_ms     время прогноза, мс (double precision), обязательный
--   actual         настоящее покрытие по словам водителя (text), пока водитель не ответил – пусто (NULL)
--   feedback_at    когда пришла обратная связь (timestamptz), может быть пустым
