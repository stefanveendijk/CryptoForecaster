@echo off
setlocal
cd /d "%~dp0"
title CryptoForecaster Update

echo.
echo === CryptoForecaster bijwerken ===
echo Deze update bewaart .venv en local_data.
echo.

if not exist start_local_app.bat (
  echo FOUT: voer dit bestand uit vanuit de bestaande CryptoForecaster-map.
  pause
  exit /b 1
)

echo Lokale server wordt indien nodig gestopt...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080" ^| findstr "LISTENING"') do (
  taskkill /PID %%a /F >nul 2>nul
)

set "ROOT=%CD%"
set "CFTMP=%TEMP%\CryptoForecasterUpdate"
if exist "%CFTMP%" rmdir /s /q "%CFTMP%"
mkdir "%CFTMP%" >nul 2>nul

echo Nieuwste versie wordt opgehaald van GitHub...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$zip=Join-Path $env:CFTMP 'main.zip';" ^
  "Invoke-WebRequest -UseBasicParsing 'https://github.com/stefanveendijk/CryptoForecaster/archive/refs/heads/main.zip' -OutFile $zip;" ^
  "$unzip=Join-Path $env:CFTMP 'unzipped';" ^
  "Expand-Archive -LiteralPath $zip -DestinationPath $unzip -Force;" ^
  "$src=Get-ChildItem -LiteralPath $unzip -Directory | Select-Object -First 1;" ^
  "if(-not $src){throw 'Uitgepakte projectmap niet gevonden.'};" ^
  "if(-not (Test-Path (Join-Path $src.FullName 'local_dashboard.html'))){throw 'Dashboardbestand ontbreekt in download.'};" ^
  "robocopy $src.FullName $env:ROOT /E /XD '.git' '.venv' 'local_data' /XF 'update_local.bat' | Out-Null;" ^
  "if($LASTEXITCODE -gt 7){exit $LASTEXITCODE}"

if errorlevel 1 (
  echo.
  echo FOUT: bijwerken is mislukt. Je bestaande lokale data zijn niet verwijderd.
  pause
  exit /b 1
)

rmdir /s /q "%CFTMP%" >nul 2>nul

echo.
echo UPDATE GESLAAGD.
echo De bestaande .venv en local_data zijn behouden.
echo De lokale app wordt nu opnieuw gestart.
echo.
call start_local_app.bat
