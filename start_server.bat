@echo off
chcp 65001 >nul
title OnyX AI - Python Core Server
echo.
echo  ============================================
echo   OnyX AI v0.5 - Python Core Server
echo  ============================================
echo.
echo  Запускаю локальный Python-сервер (http://127.0.0.1:8787)
echo  Не закрывай это окно, пока пользуешься OnyX.
echo.
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
cd /d "%~dp0"
python -X utf8 python_core\server.py 8787
echo.
echo  Сервер остановлен.
pause