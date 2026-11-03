@echo off
title GodsEye - environment check
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

%PY% demo_check.py
echo.
pause
goto :eof

:nopython
echo.
echo   No Python 3.11+ found on this machine.
echo   Install it from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH", then run this again.
echo.
pause
