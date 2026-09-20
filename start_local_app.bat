@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Eerst install_windows_local.bat uitvoeren.
  pause
  exit /b 1
)
if not exist local_data mkdir local_data
set DATA_DIR=%CD%\local_data
set AUTO_RUN_MODEL=true
set MODEL_HOUR_UTC=4
set MODEL_MINUTE_UTC=15
set MODEL_NEWS=true
set MODEL_DEEP=false
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:8080/api/v1/dashboard
python -m uvicorn cloud_server:app --host 127.0.0.1 --port 8080
