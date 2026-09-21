@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title CryptoForecaster Update

echo.
echo === CryptoForecaster automatisch bijwerken ===
echo .venv en local_data blijven behouden.
echo.

if not exist "start_local_app.bat" (
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
set "ZIP=%CFTMP%\main.zip"
set "UNZIP=%CFTMP%\unzipped"
set "SRC=%UNZIP%\CryptoForecaster-main"

if exist "%CFTMP%" rmdir /s /q "%CFTMP%"
mkdir "%CFTMP%" >nul 2>nul

echo Nieuwste versie wordt opgehaald van GitHub...
curl.exe -L --fail --retry 2 --connect-timeout 20 ^
  -o "%ZIP%" ^
  "https://codeload.github.com/stefanveendijk/CryptoForecaster/zip/refs/heads/main"
if errorlevel 1 goto :fail

echo Bestanden worden uitgepakt...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '%ZIP%' -DestinationPath '%UNZIP%' -Force"
if errorlevel 1 goto :fail

if not exist "%SRC%\local_dashboard.html" (
  echo FOUT: de gedownloade versie is niet compleet.
  goto :fail
)

echo Appbestanden worden bijgewerkt...
robocopy "%SRC%" "%ROOT%" /E /R:2 /W:1 ^
  /XD ".git" ".venv" "local_data" ^
  /XF "update_local.bat" >nul
set "RC=%ERRORLEVEL%"

rem Robocopy codes 0 t/m 7 zijn succesvol; 8 of hoger is echt een fout.
if %RC% GEQ 8 goto :fail

if exist "%CFTMP%" rmdir /s /q "%CFTMP%" >nul 2>nul

echo.
echo UPDATE GESLAAGD.
echo .venv en local_data zijn behouden.
echo De lokale app wordt nu gestart.
echo.
call start_local_app.bat
exit /b 0

:fail
echo.
echo FOUT: bijwerken is mislukt.
echo Je bestaande .venv en local_data zijn niet verwijderd.
echo.
pause
exit /b 1
