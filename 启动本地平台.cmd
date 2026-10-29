@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0demos\live-platform\Open-LivePlatform.ps1" %*
if errorlevel 1 pause
