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

powershell -NoProfile -ExecutionPolicy Bypass -Command "$code='[DllImport(\"kernel32.dll\")]public static extern uint SetThreadExecutionState(uint esFlags);'; Add-Type -MemberDefinition $code -Name Power -Namespace Win32; [Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null; & '.\.venv\Scripts\python.exe' 'crypto_forecaster_ultimate.py' '--workdir' \"$env:DATA_DIR\ultimate_run\"; $rc=$LASTEXITCODE; [Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null; exit $rc"
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
