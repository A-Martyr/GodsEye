@echo off
rem ======================================================================
rem  Live webcam plate reader - browser page on http://localhost:8503
rem
rem  Same engine and the same annotated frame as livecam.bat; this one
rem  streams it as MJPEG and adds a selectable-text readout, which is what
rem  makes it legible on a projector. Its own port and its own process, so
rem  a stalled webcam cannot take the control room's API down with it.
rem ======================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv-paddle\Scripts\python.exe" (
  echo [livecam] .venv-paddle is missing - see PADDLE_SETUP.md.
  call "%~dp0_env.bat"
  if not defined PY ( echo [livecam] No Python 3.11+ found. & exit /b 1 )
  %PY% -m livecam.server %*
  exit /b %errorlevel%
)

echo [livecam] http://localhost:8503
".venv-paddle\Scripts\python.exe" -m livecam.server %*
endlocal
