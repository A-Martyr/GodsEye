@echo off
rem Rebuilds the demo database from scratch - about a minute. Wipes what is
rem there now; seed.py --keep appends instead, if that is what you wanted.
title GodsEye - seed the database
cd /d "%~dp0"
call "%~dp0_env.bat"
if not defined PY goto nopython

echo.
echo   Generating about six hours of city traffic. Takes roughly a minute.
echo.
%PY% seed.py %*
echo.
pause
goto :eof

:nopython
echo.
echo   No Python 3.11+ found. Install it and tick "Add python.exe to PATH".
echo.
pause
