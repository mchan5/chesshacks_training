@echo off
REM Auto-sync script for Windows
REM Runs the Python auto-sync script in loop mode

echo Starting auto-sync (every 1 hour)...
echo Press Ctrl+C to stop
echo.

python auto_sync.py --loop
