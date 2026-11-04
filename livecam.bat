@echo off
rem ======================================================================
rem  Live webcam plate reader - native window (lowest latency).
rem
rem  Runs from .venv-paddle, not the main interpreter: the anpr_ocr reader
rem  is the only backend with real-world accuracy (91.0% on real
rem  photographs against the CRNN's 0.9%), a printed board in front of a
rem  webcam IS a real photograph, and paddle has no wheels for the 3.13
rem  the rest of the platform runs on.
rem
rem  Keys: q quit - f fullscreen - space pause - r reset - e enrol
rem ======================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv-paddle\Scripts\python.exe" (
  echo [livecam] .venv-paddle is missing - see PADDLE_SETUP.md.
  echo [livecam] Falling back to the CRNN, which reads real boards poorly.
  call "%~dp0_env.bat"
  if not defined PY ( echo [livecam] No Python 3.11+ found. & exit /b 1 )
  %PY% -m livecam.window %*
  exit /b %errorlevel%
)

rem Only one CUDA context at a time on this card: sharing it with an API that
rem is also on the GPU starves both by roughly 50x. See RUNBOOK.md.
echo [livecam] If the API is running, start it with GODSEYE_DEVICE=cpu so this reader gets the GPU.
".venv-paddle\Scripts\python.exe" -m livecam.window %*
endlocal
