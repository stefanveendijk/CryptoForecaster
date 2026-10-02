@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe exit /b 1
if not exist local_data mkdir local_data
set DATA_DIR=%CD%\local_data
set MODEL_NEWS=true
set MODEL_DEEP=false
set LOG=%DATA_DIR%\daily_model.log

echo.>> "%LOG%"
echo ============================================================>> "%LOG%"
echo START %DATE% %TIME%>> "%LOG%"

rem Voorkom slaapstand zolang de modelrun actief is.
".venv\Scripts\python.exe" keep_awake_run.py --workdir "%DATA_DIR%\ultimate_run" >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%

if "%RC%"=="0" (
  echo MODEL SUCCES %DATE% %TIME%>> "%LOG%"
  rem Extra vangnet: controleer direct na elke geslaagde dagelijkse run of de
  rem uitgifte ook werkelijk in de live bewijslaag staat en beoordeel rijpe regels.
  if exist repair_proof_local.py (
    ".venv\Scripts\python.exe" repair_proof_local.py "%DATA_DIR%" >> "%LOG%" 2>&1
    set PRC=%ERRORLEVEL%
    if not "%PRC%"=="0" echo BEWIJSLAAG WAARSCHUWING exitcode=%PRC% %DATE% %TIME%>> "%LOG%"
  )
  echo SUCCES %DATE% %TIME%>> "%LOG%"
) else (
  echo FOUT exitcode=%RC% %DATE% %TIME%>> "%LOG%"
)
exit /b %RC%
