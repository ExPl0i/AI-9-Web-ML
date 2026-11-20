@echo off
chcp 65001 > nul
rem Т4.Б1.4, вариант А. Консольный клиент psql от имени pokrytie в базе pokrytie.
rem Дополнительные параметры передаются psql, например: db\db_psql.bat -c "SELECT * FROM predictions"
call "%~dp0db_env.bat"
"%PGBIN%\psql" -h 127.0.0.1 -p %PGPORT% -U pokrytie -d pokrytie %*
