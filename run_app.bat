@echo off
setlocal
chcp 65001 >NUL

rem Переходим в папку с батником
pushd "%~dp0"

rem Если не используете .env — можно задать токены здесь (раскомментируйте и заполните):
rem set TELEGRAM_BOT_TOKEN=ваш_тг_токен
rem set OPENAI_API_KEY=ваш_openai_key

rem ВАЖНО: Устанавливаем правильную модель (gpt-5 НЕ существует!)
set OPENAI_MODEL=gpt-4o-mini

rem Активируем локальное виртуальное окружение, если есть
if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
)

rem Запускаем бота
python architecturebot\main.py
set "RC=%ERRORLEVEL%"

popd
exit /b %RC%