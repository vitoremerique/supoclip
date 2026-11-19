@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Postador TikTok - SupoClip
echo Postador do TikTok do SupoClip (agenda os cortes para daqui 15 min)
echo Deixe esta janela aberta enquanto agenda os cortes pelo site.
echo.
"postador\.venv\Scripts\python.exe" postador\servidor.py %*
pause
