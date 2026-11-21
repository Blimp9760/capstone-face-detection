@echo off
cd /d "%~dp0"
py -3 run.py
if errorlevel 1 (
  echo.
  echo Setup failed; see the error messages above.
  pause
)
