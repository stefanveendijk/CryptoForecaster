@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CryptoForecaster dagelijkse modelrun instellen

set "TASKNAME=CryptoForecaster Daily"
set "RUNNER=%CD%\run_daily_local.bat"

if not exist "%RUNNER%" (
  echo FOUT: run_daily_local.bat ontbreekt.
  exit /b 1
)

echo Dagelijkse CryptoForecaster-run wordt ingesteld op 06:30...

schtasks /Create /TN "%TASKNAME%" /TR "\"%RUNNER%\"" /SC DAILY /ST 06:30 /F /RL LIMITED >nul
if errorlevel 1 (
  echo FOUT: Windows Taakplanner kon de dagelijkse taak niet aanmaken.
  echo Probeer dit bestand eventueel eenmaal met rechtermuisknop ^> Als administrator uitvoeren.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$s=New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew; Set-ScheduledTask -TaskName '%TASKNAME%' -Settings $s | Out-Null" >nul 2>nul

if errorlevel 1 (
  echo WAARSCHUWING: de taak staat op 06:30, maar WakeToRun/StartWhenAvailable kon niet automatisch worden ingesteld.
  echo De dagelijkse taak zelf is wel aangemaakt.
  exit /b 0
)

echo GEREED: dagelijkse modelrun om 06:30.
echo Als 06:30 wordt gemist doordat Windows niet beschikbaar is, start de taak zodra dat weer kan.
exit /b 0
