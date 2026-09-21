@echo off
setlocal
cd /d "%~dp0"
title CryptoForecaster Local Server
if not exist .venv\Scripts\python.exe (
  echo FOUT: lokale installatie ontbreekt.
  echo Voer eerst install_windows_local.bat uit.
  pause
  exit /b 1
)
if not exist local_data mkdir local_data
set DATA_DIR=%CD%\local_data
set AUTO_RUN_MODEL=false
set LOCAL_MODE=true
set MODEL_NEWS=true
set MODEL_DEEP=false
echo.
echo Oude lokale server op poort 8080 wordt indien nodig afgesloten...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING"') do (
  taskkill /PID %%a /F >nul 2>nul
)
timeout /t 1 /nobreak >nul
echo.
echo CryptoForecaster start lokaal op http://127.0.0.1:8080
echo Laat dit venster open zolang je de lokale app gebruikt.
echo.
start "" http://127.0.0.1:8080/
".venv\Scripts\python.exe" -m uvicorn cloud_server:app --host 127.0.0.1 --port 8080
echo.
echo De lokale server is gestopt of kon niet starten.
pause
