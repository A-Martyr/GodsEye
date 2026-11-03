@echo off
rem ======================================================================
rem  Shared by every launcher here. Not meant to be run on its own.
rem
rem  Leaves a working interpreter in %PY%. Deliberately does NOT use the
rem  bundled .venv folders: their pyvenv.cfg points at the machine they
rem  were built on, so activating one fails with a missing-python error
rem  that looks like a broken bundle. The py launcher is tried first
rem  because it finds a real 3.13 even when PATH points at the Microsoft
rem  Store stub.
rem ======================================================================
set "PY="
call :probe py -3.13
if not defined PY call :probe py -3
if not defined PY call :probe python
goto :eof

:probe
if defined PY goto :eof
%* -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%*"
goto :eof
