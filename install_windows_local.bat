@echo off
setlocal
cd /d "%~dp0"
echo === CryptoForecaster lokale installatie ===
where py >nul 2>nul || (echo Python ontbreekt. Installeer Python 3.12 vanaf python.org en vink "Add Python to PATH" aan.& pause & exit /b 1)
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist local_data mkdir local_data
set "TASKCMD=cmd /c \"\"%CD%\run_daily_local.bat\"\""
schtasks /Create /TN "CryptoForecaster Daily" /TR "%TASKCMD%" /SC DAILY /ST 05:30 /F
echo.
echo Installatie klaar. Dagelijkse modelrun: 05:30 lokale Windows-tijd.
echo Start nu start_local_app.bat
pause
