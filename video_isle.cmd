@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo MP4 dosyanizi bu dosyanin uzerine surukleyip birakin.
  pause
  exit /b 1
)
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
"%~dp0.venv\Scripts\python.exe" -u -m src.drowning_demo.swimmers %*
if errorlevel 1 (
  echo Video islenemedi. Yukaridaki hata mesajini kontrol edin.
  pause
  exit /b 1
)
echo Hazir. Yukarida yazan annotated_video.mp4 dosyasini acabilirsiniz.
pause
