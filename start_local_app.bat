@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CryptoForecaster Local

if not exist ".venv\Scripts\python.exe" (
  echo FOUT: lokale installatie ontbreekt.
  echo Voer eerst install_windows_local.bat uit.
  pause
  exit /b 1
)

if not exist "local_dashboard.html" (
  echo FOUT: local_dashboard.html ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist "trading_strategy.py" (
  echo FOUT: trading_strategy.py ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist "strategy_audit.py" (
  echo FOUT: strategy_audit.py ontbreekt. Werk de app eerst bij.
  pause
  exit /b 1
)

if not exist local_data mkdir local_data
set "DATA_DIR=%CD%\local_data"
set "AUTO_RUN_MODEL=false"
set "LOCAL_MODE=true"
set "MODEL_NEWS=true"
set "MODEL_DEEP=false"

echo.
echo Oude lokale server op poort 8080 wordt indien nodig afgesloten...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING"') do (
  taskkill /PID %%a /F >nul 2>nul
)
timeout /t 1 /nobreak >nul

echo Nieuwe lokale server wordt gestart...
start "CryptoForecaster Server" /min cmd /k ""%CD%\.venv\Scripts\python.exe" -m uvicorn cloud_server:app --host 127.0.0.1 --port 8080"

echo Wachten totdat de server gereed is...
for /l %%i in (1,1,30) do (
  curl.exe -s --fail "http://127.0.0.1:8080/api/v1/health" >nul 2>nul
  if not errorlevel 1 goto :ready
  timeout /t 1 /nobreak >nul
)

echo.
echo FOUT: de lokale server antwoordt niet binnen 30 seconden.
echo Open het geminimaliseerde venster "CryptoForecaster Server" voor de foutmelding.
pause
exit /b 1

:ready
echo Server gereed.
echo Dashboard wordt geopend op http://127.0.0.1:8080/
set "CACHEBUST=%RANDOM%%RANDOM%"
start "" "http://127.0.0.1:8080/?v=%CACHEBUST%"
exit /b 0
