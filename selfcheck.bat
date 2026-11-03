@echo off
rem The 58-check end-to-end suite. Points the database at a scratch temp
rem file, so it never touches data\godseye.db. Worth running before a demo.
title GodsEye - self check
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

%PY% tests.py
echo.
pause
goto :eof

:nopython
echo.
echo   No Python 3.11+ found. Install it and tick "Add python.exe to PATH".
echo.
pause
