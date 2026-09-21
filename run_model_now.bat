@echo off
setlocal
cd /d "%~dp0"
title CryptoForecaster Model Run
if not exist .venv\Scripts\python.exe (
  echo FOUT: lokale installatie ontbreekt.
  echo Voer eerst install_windows_local.bat uit.
  pause
  exit /b 1
)
if not exist local_data mkdir local_data
set DATA_DIR=%CD%\local_data
set MODEL_NEWS=true
set MODEL_DEEP=false
echo.
echo Handmatige CryptoForecaster modelrun wordt gestart.
echo De laptop blijft tijdens deze berekening wakker. Sluit dit venster niet.
echo.

".venv\\Scripts\\python.exe" keep_awake_run.py --workdir "%DATA_DIR%\\ultimate_run"
if errorlevel 1 (
  echo.
  echo MODEL RUN MISLUKT. Stuur een screenshot van de regels hierboven.
  pause
  exit /b 1
)
echo.
echo MODEL RUN GESLAAGD.
echo Je lokale forecasts zijn nu bijgewerkt.
pause
