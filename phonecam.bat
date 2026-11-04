@echo off
rem ======================================================================
rem  Phones as junction cameras.
rem
rem    phonecam.bat                              default 3 junctions
rem    phonecam.bat --junctions CAM-20,CAM-21    pick your own
rem    phonecam.bat --realtime                   demo the clone detector
rem    phonecam.bat --no-enrol                   read only, touch no data
rem
rem  Runs from .venv-paddle: anpr_ocr is the only backend with real-world
rem  accuracy, and a phone camera pointed at a board is a real photograph.
rem
rem  Serves https with a self-signed certificate, because a browser will
rem  not hand a page its camera over plain http. The phones warn once.
rem ======================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv-paddle\Scripts\python.exe" (
  echo [phonecam] .venv-paddle is missing - see PADDLE_SETUP.md.
  echo [phonecam] Falling back to the CRNN, which reads real plates poorly.
  call "%~dp0_env.bat"
  if not defined PY ( echo [phonecam] No Python 3.11+ found. & exit /b 1 )
  %PY% -m phonecam.server %*
  exit /b %errorlevel%
)

rem One CUDA context at a time on this card - sharing it with an API that is
rem also on the GPU starves both by roughly 50x. See RUNBOOK.md.
echo [phonecam] If the API is running, start it with GODSEYE_DEVICE=cpu so this reader gets the GPU.
echo [phonecam] Phones must be on the same Wi-Fi as this laptop.
".venv-paddle\Scripts\python.exe" -m phonecam.server %*
endlocal
