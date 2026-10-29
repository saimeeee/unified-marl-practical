@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0demos\live-platform\Stop-LivePlatform.ps1" %*
if errorlevel 1 pause
